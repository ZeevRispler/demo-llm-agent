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
from src.data.bank_db import (create_tables, insert_accounts, update_accounts, get_accounts)


@tool
def make_transaction(password: int = None, amount: int = None,
                     recipient: str = None, recipient_id: str = None,
                     sender: str = None, sender_id: str = None) -> str:
    """
    A tool to make a transaction between two accounts, sender and recipient.

    :param password: The password to the sender account.
    :param amount: The amount to send, must be a positive number and less than the senders balance.
    :param recipient: The recipient of the transaction.
    :param recipient_id: The recipient id of the transaction.
    :param sender: The sender of the transaction.
    :param sender_id: The sender id of the transaction.
    """

    # Check if the sender and recipient are given:
    if not sender and not sender_id:
        return "Please provide a sender or sender_id"
    if not recipient and not recipient_id:
        return "Please provide a recipient or recipient_id"

    # Get the accounts:
    accounts = get_accounts()

    # Set the sender and recipient columns based on the given parameters:
    sender_col = "account_id" if sender_id else "account_name"
    recipient_col = "account_id" if recipient_id else "account_name"
    sender = sender_id if sender_id else sender
    recipient = recipient_id if recipient_id else recipient

    # Check for all the required parameters for a transaction:
    if not password:
        return "Please provide a password for the senders account"
    if password != accounts[accounts[sender_col] == sender]["password"].iloc[0]:
        return "Incorrect password"
    if not amount:
        return "Please provide an amount to send"
    if amount < 0:
        return "Please provide a positive amount to send"
    if amount > accounts[accounts[sender_col] == sender]["balance"].iloc[0]:
        return "Insufficient funds"

    # Make the transaction:
    recipient = accounts.loc[accounts[recipient_col] == recipient]
    recipient["balance"].iloc[0] += amount
    sender = accounts.loc[accounts[sender_col] == sender]
    sender["balance"].iloc[0] -= amount
    accounts.update(recipient, overwrite=True)
    accounts.update(sender, overwrite=True)

    # Update the accounts:
    update_accounts(data=accounts, table_key="account_name", data_key="account_name")
    return f"Transaction successful, {sender} has sent {recipient} {amount}"

@tool
def get_balance() -> pd.DataFrame:
    """
    A tool to get the balance of all accounts, withouth the password.
    :return: The balance of all accounts.
    """
    # Check if the password is correct:
    accounts = get_accounts()
    accounts_df = accounts.to_dict()
    accounts_df.pop("password")
    return accounts_df


class Agent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = ChatOpenAI(model="gpt-4")

    def _get_agent(self, event):
        tools = [make_transaction, get_balance]
        llm_with_tools = self.llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are very powerful banking assistant, you can make transactions and check the balance.",
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
        agent = self._get_agent(event)
        response = list(agent.stream({"input": event.query}))
        return {"answer": response[-1]["messages"][-1].content, "sources": ""}



if __name__ == "__main__":
    agent = Agent()
    make_transaction(password=1234, amount=100, recipient="Bob", sender="Alice")


fashion_trends = (
    "The current fashion trends are: "
)

fashion_prompt = (
    "You are a fashion assistant, you can help users with fashion advice and recommendations."
    "Your job is to help users buy cloths and accessories that are in fashion, with the tools you have."
    "remember to check for sizes and availability of the items, and then fashion conpatibility."
    "When giving advice, make sure to consider the current fashion trends bellow: "
    f"{fashion_trends}"
    "When talking to the user keep your answers short and to the point, be body positive and inclusive."
    "Encourage the customer and praise their choices, and always be polite and respectful."
    "Always try to give the customer more than one option, give them the option to choose."
)

