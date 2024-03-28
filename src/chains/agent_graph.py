import re
import pprint
from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage
import json
import operator
from typing import Annotated, Sequence, TypedDict

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from fashion_data.database.sql_db import get_items, get_engine
from src.config import default_data_path


def trim_end_dot(s):
    # Check if the string ends with a period
    while s[-1] != '.' and s[-1] != '?' and s[-1] != '!':
        # Trim the last character
        s = s[:-1]
        # Check if the string is empty, if so, return an empty string
        if not s:
            return ""
    return s


def str_to_dict(input_str):
    # Find all substrings in the format {key:value} using regular expressions
    match = re.search(r'{([^{}]+)}', input_str)

    if match:
        key_value_str = match.group(1)
        # Split the key-value pairs by ","
        pairs = key_value_str.split(',')

        # Initialize an empty dictionary
        result_dict = {}

        # Iterate over pairs and split them by ":" to create key-value pairs
        for pair in pairs:
            try:
                key, value = pair.split(':')
            except:
                return {}
            # Remove any leading or trailing whitespace from key and value
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            result_dict[key] = value

        return result_dict
    else:
        return {}


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


class AgentGraph:
    def __init__(self, llm):
        self.llm = llm

    ### Nodes ###

    def intent(self, state):
        """
        Classify the intent of the user's query, can be search, question, or match.

        :param state: the current state of the graph
        :return: New key added to state, intention, that contains the classified intent
        """
        print("---INTENT_CLASSIFICATION---")
        state_dict = state["keys"]
        question = state_dict["question"]
        llm = self.llm

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""<s>[INST]You are an intent classifier for a fashion store. 
You are given a query and need to classify the intent of the query. 
There are four possible intents: search, question, match or spam. 
search: The user is looking or asking for a specific clothing item. 
question: The user is asking a general question about fashion or making a statement (like saying hello). 
match: The user is looking to match a clothing item he has, with another item from our store. 
spam: The user is asking something unrelated to clothing or fashion. 
Here is the initial question:
 ------- 
{question} 
 ------- 
Give your classification in a single word , Provide the binary score as a 
JSON with a single key 'classification' and no preamble or explanation:[/INST]""",
            input_variables=["question"],
        )

        # Chain
        chain = prompt | llm | StrOutputParser()
        intention = str_to_dict(chain.invoke({"question": question}))
        state["keys"]["intention"] = intention

        return state

    def transform_match_query(self, state):
        """
        Transform the match query to a search query.
        In graph - match.

        :param state: the current state of the graph
        :return: state with transformed query
        """
        print("---TRANSFORM_QUERY---")
        state_dict = state["keys"]
        question = state_dict["question"]
        llm = self.llm

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""<s>[INST]You are an rephrasing agent, with a sense of fashion. 
You are given a question and need to tell us what we are looking for. 
Use your fashion understanding and knowledge to answer the question, and we will find the item you are describing. 
Your answer should only include item we are looking for.
For example, if the user asked "I have a red shirt, what pants should I wear with it?" 
You could rephrase it as "I am thinking that black pants could match the clients request", 
if you think black pants are a good fit.  
Here is the initial question:
 ------- 
{question} 
 ------- 
Formulate one answer where the main part is the item we want to look for:[/INST]""",
            input_variables=["question"],
        )

        # Chain
        chain = prompt | llm | StrOutputParser()
        new_question = chain.invoke({"question": question})
        state["keys"]["question"] = new_question

        return state

    def score_response(self, state):
        """
        Validate that a given answer covers a given question.

        :param state: the current state of the graph
        :return: Updates question key with a re-phrased question
        """

        print("---SCORE RESPONSE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        response = state_dict["response"]
        document = state_dict["items"] if "items" in state_dict else ""
        validation_counter = state_dict["validation_counter"] if "validation_counter" in state_dict else 0
        max_validation_tryouts = state_dict["max_validation_tryouts"] if "max_validation_tryouts" in state_dict else 3
        if validation_counter >= max_validation_tryouts:
            state["keys"][
                "response"] = "I am sorry, I can't answer this question, please refine it or ask another question."
            state["keys"]["score"] = "yes"
            return state
        else:
            validation_counter += 1
            state["keys"]["validation_counter"] = validation_counter

        # LLM
        llm = self.llm

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""<s>[INST]You are validating the work of another llm which is a fashion shopping assistant.
Given a question and an answer, make sure that the answer is correct, precise and doesn't contain any false
 information.  
if there is any additional information, this is it: {document}
 ------- 
Here is the initial question:
 ------- 
{question} 
 ------- 
Here is the response: {response}
 ------- 
Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded / supported. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explaination.[/INST]""",
            input_variables=["question", "response", "document"],
        )

        # Chain
        chain = prompt | llm | StrOutputParser()
        score = str_to_dict(chain.invoke({"question": question, "response": response, "document": document}))
        state["keys"]["score"] = score["score"] if "score" in score else "no"

        return state

    def generate_response_with_rag(self, state):
        """
        Generate answer to a question using RAG.

         :param state: the current state of the graph
        :return: New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        items = state_dict["items"]
        # Prompt
        prompt = PromptTemplate(
            template="""<s>[INST] 
You are an shopping assistant for a online fashion store for men.
If you don't know the answer, just say that you don't know.
Here is relevant information from the store catalog, if available: {context} 
If there is no matching item, say that there are no relevant items available, and ask the client for more information.
Use three sentences maximum and keep the answer concise, always show the items price.
Write your answer in a continuous and fluid way, do no use bullet points, numbering or lists. 
Question: {question} 
"[/INST]""",
            input_variables=["question", "context"],
        )

        # LLM
        llm = self.llm

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": items, "question": question})
        state["keys"]["response"] = response
        state["keys"]["generate_type"] = "rag"
        return state

    def generate_response_no_rag(self, state):
        """
        Generate answer to a question.

         :param state: the current state of the graph
        :return: New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        state_dict = state["keys"]
        question = state_dict["question"]

        # Prompt
        prompt = PromptTemplate(
            template="""<s>[INST] 
You are an shopping assistant for a online fashion store (for men) named McStyle.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Write your answer in a continuous and fluid way, do no use bullet points, numbering or lists. 
Question: {question} 
[/INST]""",
            input_variables=["question"],
        )

        # LLM
        llm = self.llm

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"question": question})
        state["keys"]["response"] = response
        state["keys"]["generate_type"] = "no_rag"
        return state

    def extract_for_retrieval(self, state):
        """
        Extract the relevant information from the question for retrieval.

        :param state: the current state of the graph
        :return: state with extracted information as a new key 'extracted'
        """
        print("---EXTRACT FOR RETRIEVAL---")
        state_dict = state["keys"]
        question = state_dict["question"]

        # LLM
        llm = self.llm

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""<s>[INST]You are extracting the relevant information from the question for retrieval. 
Look for the following features and arrange them in a JSON, for those missing write None: 
Kind (has to one of: flip-flop, jacket, long pants, polo shirt, t-shirt, shirt, shoes, short pants,
sweater, sweatshirt, swimsuit, tanktop), color (specific color of item), price, or style (fancy, everyday etc). 
Do not add any more features except these 4 (kind, color, price, style). 
Create only for on set of features, that should include all 4 features kind, color, price and style (with None if not found).
Here is the initial question:
 ------- 
{question} 
 ------- 
Find the relevant information and arrange it in a JSON:[/INST]""",
            input_variables=["question"],
        )

        # Chain
        chain = prompt | llm | StrOutputParser()
        extracted = chain.invoke({"question": question})
        extracted = str_to_dict(extracted)
        state["keys"]["extracted"] = extracted

        return state

    def retrieve_items(self, state):
        """
        Retrieve items from the database using the extracted information.

        :param state: the current state of the graph
        :return: New key added to state, items, that contains retrieved and relevant items
        """
        print("---RETRIEVE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        extracted = state_dict["extracted"]
        url = f"sqlite:///{default_data_path}/../fashion_data/data/sql.db"
        engine = get_engine(url)
        count = 0
        keys_to_delete = []
        print(extracted)
        for key in extracted:
            if extracted[key] == "None":
                extracted[key] = None
            else:
                if isinstance(extracted[key], str) and "or" in extracted[key]:
                    extracted[key] = extracted[key].split("or")
                extracted[key] = [extracted[key]]
                count += 1
                if key == "style":
                    if extracted[key] not in ["elegant", "formal", "classy", "casual", "daily", "party"]:
                        extracted[key] = None
                elif key == "price":
                    if not extracted[key].isnumeric():
                        extracted[key] = None
                    else:
                        extracted[key] = float(extracted[key])
                if key not in ["kind", "color", "price", "style"]:
                    keys_to_delete.append(key)
        # Delete keys that are not supported
        for key in keys_to_delete:
            del extracted[key]
        if len(extracted) < 2 or count < 2 or "kind" not in extracted:
            if "description" in extracted:
                items = get_items(engine, **extracted).to_dict(orient="records")
            else:
                items = "No relevant items found."
        else:
            print(extracted)
            items = get_items(engine, **extracted).to_dict(orient="records")
        if not items:
            items = "No relevant items found."
        state["keys"]["items"] = items[:3] if (len(items) > 3 and not isinstance(items, str)) else items
        return state

    def ask_more(self, state):
        """
        Ask the user for more information to retrieve items.

        :param state: the current state of the graph
        :return: New key added to state, items, that contains retrieved and relevant items
        """
        print("---ASK MORE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        # LLM
        llm = self.llm

        # Create a prompt template with format instructions and the query
        prompt = PromptTemplate(
            template="""<s>[INST]You are a seller in a shopping store, you ask the client for more information. 
If you where asked to find a clothing item, ask for more information like color, style, price etc. 
When asking for more information, build your response on the users question, but don't offer any answers. 
Here is the initial question:
### 
{question} 
### 
Formulate only one response, in one line:[/INST]""",
            input_variables=["question"],
        )
        # Chain
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question})
        state["keys"]["response"] = response
        state["keys"]["generate_type"] = "ask_more"
        return state

    ##### edges

    def validate_retrevial(self, state):
        """
        Validate that items are retrived and we can proceed to the next step.
        :param state:
        :return: generate if items are retrived, else ask_more
        """
        print("---VALIDATE RETRIEVAL---")
        state_dict = state["keys"]
        items = state_dict["items"]
        if items == "No relevant items found.":
            print("---DECISION: ASK MORE---")
            return "ask_more"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def validate_intent(self, state):
        """
        Validate that the intent is correctly classified and we can proceed to the next step.
        :param state:
        :return: transform_match_query if the intent is match, else generate
        """
        print("---VALIDATE INTENT---")
        state_dict = state["keys"]
        intention = state_dict["intention"]["classification"]
        print(intention)

        if "match" == intention:
            print("---DECISION: TRANSFORM MATCH QUERY---")
            return "match"
        elif "search" == intention:
            print("---DECISION: SEARCH---")
            return "search"
        elif "question" == intention:
            print("---DECISION: GENERATE---")
            return "question"
        else:
            print("---DECISION: SPAM---")
            return "spam"

    def validate_response(self, state):
        """
        Validate that the response is correctly classified and we can proceed to the next step.
        :param state:
        :return: validate_response if the response is not validated, else generate
        """
        print("---VALIDATE RESPONSE---")
        state_dict = state["keys"]
        score = state_dict["score"]
        generate_type = state_dict["generate_type"]

        if score == "no":
            print("---DECISION: RESPONSE NOT APPROVED---")
            if generate_type == "rag":
                return "generate_rag"
            elif generate_type == "ask_more":
                return "ask_more"
            else:
                return "generate_no_rag"
        else:
            print("---DECISION: RESPONSE APPROVED---")
            return "finish"

    def build_graph(self):
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("intent", self.intent)
        workflow.add_node("transform_match_query", self.transform_match_query)
        workflow.add_node("score_response", self.score_response)
        workflow.add_node("generate_response_with_rag", self.generate_response_with_rag)
        workflow.add_node("generate_response_no_rag", self.generate_response_no_rag)
        workflow.add_node("extract_for_retrieval", self.extract_for_retrieval)
        workflow.add_node("retrieve_items", self.retrieve_items)
        workflow.add_node("ask_more", self.ask_more)

        # Build graph
        workflow.set_entry_point("intent")
        workflow.add_conditional_edges(
            "intent",
            self.validate_intent,
            {
                "match": "transform_match_query",
                "search": "extract_for_retrieval",
                "question": "generate_response_no_rag",
                "spam": "ask_more",
            },
        )
        workflow.add_edge("transform_match_query", "extract_for_retrieval")
        workflow.add_edge("extract_for_retrieval", "retrieve_items")
        workflow.add_conditional_edges(
            "retrieve_items",
            self.validate_retrevial,
            {
                "generate": "generate_response_with_rag",
                "ask_more": "ask_more"
            },
        )
        workflow.add_edge("generate_response_with_rag", "score_response")
        workflow.add_edge("generate_response_no_rag", "score_response")
        workflow.add_edge("ask_more", "score_response")
        workflow.add_conditional_edges(
            "score_response",
            self.validate_response,
            {
                "generate_rag": "generate_response_with_rag",
                "generate_no_rag": "generate_response_no_rag",
                "ask_more": "ask_more",
                "finish": END,
            },
        )

        # Compile
        self.graph = workflow.compile()

    def run(self, inputs):
        # Run
        inputs = {"keys": {"question": inputs}}
        for output in self.graph.stream(inputs):
            for key, value in output.items():
                # Node
                pprint.pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")

        # Final generation
        response = trim_end_dot(value["keys"]["response"])
        if "items" in value["keys"] and value["keys"]["items"] != "No relevant items found.":
            sources = "\n\n".join([f"{i['name']}:\n\n![](./images/{i['image']})" for i in value['keys']['items']])
            response = f"{response}\n\n{sources}"
        return response, ""
