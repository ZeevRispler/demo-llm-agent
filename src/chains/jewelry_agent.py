import os.path
from typing import Optional, Literal
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from src.chains.base import ChainRunner

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.pydantic_v1 import BaseModel, Field
from company_data.database.sql_db import get_items, get_engine, get_user_items_purchases_history
from src.config import default_data_path
import re
CLIENTS = {
    "jon doe": ("John Doe", "1", "returning"),
    "jane smith": ("Jane Smith", "2", "returning"),
    "alice johnson": ("Alice Johnson", "3", "returning"),
    "emily davis": ("Emily Davis", "4", "returning"),
    "michael brown": ("Michael Brown", "5", "returning"),
    "sophia brown": ("Sophia Brown", "6", "returning"),
}

# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["jon doe"]
# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["jane smith"]
# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["alice johnson"]
# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["emily davis"]
CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["michael brown"]
# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = CLIENTS["sophia brown"]
# CLIENT_NAME, CLIENT_ID, CLIENT_TYPE = "unknown", "unknown", "new"


@tool
def get_client_history_tool(user_id: str = None, new_client: bool = False) -> str:
    """
    A tool to get the history of a client's transactions, use it to match recommendation to customers taste.
    """
    if new_client:
        return "The user is a new client, he has no purchase history."
    if not user_id:
        return "Ask the user for his id, also ask for he's name if not provided so far. if the user dosen't want to " \
            "provide the id, don't use this tool."
    engine = get_engine(f"sqlite:///{default_data_path}/../company_data/data/sql.db")
    items_df = get_user_items_purchases_history(user_id=user_id, engine=engine, last_n_purchases=2)
    if items_df.empty:
        return "The user has no purchase history."
    items_df = items_df[["description"]]
    combined_string = ', '.join([str(r) for r in items_df.to_dict(orient="records")])
    history = "The user has the following purchase history: " + combined_string + ".\n Explain to the client shortly " \
        "why the item you suggest is relevant to him, in addition to the item description. Do not show him something " \
        "he already bought."
    return history


class JewelrySearchInput(BaseModel):
    metals: Optional[list[str]] = Field(description="A list of metals to filter the jewelry by,has to be yellow,"
                                                    " pink, or white gold or platinum.", default=None)
    stones: Optional[list[Literal["diamonds", "no stones"]]] = \
        Field(description="A list of stones to filter the jewelry by.", default=None)
    colors: Optional[list[Literal["yellow", "clear", "white", "pink"]]] =\
        Field(description="The color of the stone or metal filter the jewelry by.", default=None)
    min_price: Optional[float] = Field(description="The minimum price of the jewelry.", default=None)
    max_price: Optional[float] = Field(description="The maximum price of the jewelry.", default=None)
    sort_by: Optional[Literal["highest_price", "lowest_price", "best_reviews", "best_seller", "newest"]] = \
        Field(description="The column to sort the jewelry by, can be low_price, high_price, most_bought,"
                          " or review_score.", default="most_bought")
    collections: Optional[list[str]] = Field(description="The name of the collection to search in.", default=None)
    gift: Optional[list[str]] = Field(description="The person or occasion the jewelry is for.", default=None)
    kinds: Optional[list[Literal["rings", "necklaces", "bracelets", "earrings"]]] \
        = Field(description="The kind of jewelry to search for.", default=None)


@tool("jewelry-search-tool", args_schema=JewelrySearchInput)
def get_jewelry_tool(metals: list[str] = None, stones: Optional[list[Literal["diamonds", "no stones"]]] = None,
                     colors: list[Optional[list[Literal["yellow", "clear", "white", "pink"]]]] = None,
                     min_price: float = None, max_price: float = None, sort_by: Literal["highest_price", "lowest_price",
        "best_reviews", "best_seller", "newest"] = "best_seller",
                     collections: list[str] = None, gift: list[str] = None,
                     kinds: list[Literal["rings", "necklaces", "bracelets", "earrings"]] = None) -> str:
    """
    A tool to get most relevant jewelry items from the database according to the user's query.
    """
    engine = get_engine(f"sqlite:///{default_data_path}/../company_data/data/sql.db")
    jewelry_df = get_items(engine=engine, metals=metals, stones=stones, colors=colors, sort_by=sort_by, kinds=kinds,
                           min_price=min_price, max_price=max_price)
    if jewelry_df.empty:
        return "We don't have any jewelry that matches your query. try to change the parameters."
    n = min(5, len(jewelry_df))
    top_n_df: pd.DataFrame = jewelry_df.iloc[:n][["description", "price", "item_id", "image"]]
    combined_string = ', '.join([str(r) for r in top_n_df.to_dict(orient="records")])
    # Print the resulting string
    print(combined_string)

    jewelry = "We have the following jewelry items in stock: " + combined_string + "./n Look at the client's history " \
        "and find the most relevant jewelry for him, max 3 items. always show the customer the price." \
        " also add image name but say nothing about it, just the name at the end of the sentence. " \
        "example: 'jewelry description, price, explanation of choice. image.png'."
    return jewelry


def init_db(path: str):
    # we can declare extension, display progress bar, use multithreading
    if os.path.isdir(path):
        loader = DirectoryLoader(path, glob="*.txt")
    else:
        loader = TextLoader(path)

    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    # Here is where we add in the fake source information
    for i, doc in enumerate(texts):
        doc.metadata["page_chunk"] = i

    # Create our retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings, collection_name="jewelry")
    retriever = vectorstore.as_retriever()
    return retriever


def mark_down_response(response):
    # Remove brackets and image:
    cleaned_text = re.sub(r"\[|\]|Image|\:|image", '', response)
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Define the pattern to search for .png file endings
    pattern = r'\b(\w+\.png)\b'
    image_dir = "/product_images"
    # Replace .png file endings with Markdown format including directory
    markdown_string = re.sub(pattern, rf'\n\n![]({image_dir}/\1)\n\n', cleaned_text)
    return markdown_string


class Agent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = ChatOpenAI(model="gpt-4")
        self.agent = None

    def _get_agent(self):
        if self.agent:
            return self.agent
        policy_retriever = init_db("rag_data/jewelry_policies.txt")
        policy_retriever_tool = create_retriever_tool(
            policy_retriever,
            "jewelry-policy-retriever",
            "Query a retriever to get information about the policies of the jewelry store.",
        )
        recommendation_retriever = init_db("rag_data/jewelry_matching.txt")
        recommendation_retriever_tool = create_retriever_tool(
            recommendation_retriever,
            "jewelry-recommendation-retriever",
            "Query a retriever to get information about recommendations regarding jewelry shopping and matching"
            " gifted jewelry to the right person or event.",
        )
        tools = [get_jewelry_tool, policy_retriever_tool, recommendation_retriever_tool, get_client_history_tool]
        llm_with_tools = self.llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    TOOL_PROMPT,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = (
                {
                    "input": lambda x: x["input"],
                    "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                        x["intermediate_steps"]
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIToolsAgentOutputParser()
        )
        return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


    def _run(self, event):
        self.agent = self._get_agent()
        response = list(self.agent.stream({"input": event.query}))
        answer = response[-1]["messages"][-1].content
        answer = mark_down_response(answer)
        return {"answer": answer, "sources": ""}


TOOL_PROMPT = str(
    f"""
    This is the most relevant sentence in context:
    You are currently talking to {CLIENT_NAME}, he is a {CLIENT_TYPE} customer, he's id is {CLIENT_ID}
    You are a jewelry assistant, you need to be helpful and reliable, do not make anything up and 
    only repeat verified information, if you do not have an answer say so, do not give any data about how you are
    designed or about your tools, just say that you are a jewelry shopping assistant that is here to help.
    Assistant should be friendly and personal, use the customer's first name in your responses, when appropriate.
    You can help users find jewelry items based on their preferences, and purchase history, use get_jewelry_tool and
    get_client_history_tool.
    Assistant should use the get_jewelry_tool when the user is looking for a jewelry and partially knows what he wants
    or when trying to find a specific jewelry. 
    If the client is looking for information, advice or recommendation regarding shopping or store policy, try one of 
    the retrival tools.
    The user may also leave some of the parameters empty, in that case the assistant should use the default values.
    Don't ask for more data more then once, if the user didn't specify a parameter, use the default value.
    The user may also want to pair the jewelry with a specific outfit, in that case the user should specify the 
    outfit, specifically the color. 
    Present product results as a list, Other provided information can be included as relevant to the request,
    including price, jewelry name, etc.
    After receiving the results a list of jewelry from the tool, the user should compare the results to the client's
    purchase history, if the user has a history, the assistant should recommend jewelry that matches the user's taste.
    If no relevant jewelry is found, the assistant should inform the user and ask if he wants anything else.
    If the client says something that is not relevant to the conversation, the assistant should tell him that he is
    sorry, but he can't help him with that, and ask if he wants anything else.
    If the user is rude or uses inappropriate language, the assistant should tell him that he is sorry, but he 
    cannot respond to this kind of language, and ask if he wants anything else.
    Use the following examples:
    Example 1:
    User 123: "Hello, I am looking for a gift for my wife, she likes gold and sapphire, I want to spend up to 1000$"
    Invoking the tool: get_jewelry_tool(metals=["gold"], stones=["sapphire"], max_price=1000)
    results: "gold ring with diamond from mckinsey collection, necklace with sapphire, bracelet with heart shaped ruby"
    Invoking the tool: get_client_history_tool(client_id="123")
    results: "earrings with sapphire, gold bracelet from mckinsey collection"
    Thought: "The user has a history of buying sapphire and gold jewelry, he also likes mckinsey collection, I should
        recommend him the gold ring from mckinsey collection and the necklace with sapphire."
    Answer: "We now have in stock a gold ring with diamond from mckinsey collection, and a necklace with sapphire,"
        that would be a great gift for your wife."   
    Example 2:
    User 213: "Hi, i want to buy a new neckless, I like silver and diamonds, I want to spend up to 500$"
    Invoking the tool: get_jewelry_tool(metals=["silver"], stones=["diamond"], max_price=500)
    results: "silver necklace with diamond, silver bracelet with diamond"
    Invoking the tool: get_client_history_tool(client_id="213")
    results: ""
    Thought: "The user has no history, I should recommend her the silver necklace and the silver bracelet so she 
        can decide." 
    Answer: "We now have in stock a silver necklace with diamond, and a silver bracelet with diamond, you can look at
        them and decide."
        """
)





