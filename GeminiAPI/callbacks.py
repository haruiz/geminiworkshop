from typing import Any, Dict, List

from dotenv import load_dotenv, find_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(find_dotenv())


def get_chat_history():
    """
    This function returns the chat history.
    :param history:
    :return:
    """
    return [
        HumanMessage("What is the capital of France?"),
        AIMessage("The capital of France is Paris."),
    ]


class PromptLogger(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        print(f"this is the prompt generated: \n {formatted_prompts}")


llm = ChatGoogleGenerativeAI(model="gemini-pro")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
chat_history = RunnablePassthrough.assign(history=lambda _: get_chat_history())
chain = chat_history | prompt | llm | {"answer": StrOutputParser()}
response = chain.invoke(
    {
        "question": "Can you respond again?",
    },
    config={"callbacks": [PromptLogger()]},
)
print(response)
