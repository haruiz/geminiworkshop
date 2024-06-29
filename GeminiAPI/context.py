from operator import itemgetter

from dotenv import load_dotenv, find_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    GoogleGenerativeAI,
)
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(find_dotenv())


def no_memory_interaction():
    """
    This function shows an example of what a chat app would look like without memory.
    :return:
    """
    template = (
        "you are a chatbot having a conversation with a human, answer the following question, \n"
        "{question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = prompt | llm | StrOutputParser()
    ai_message = chain.invoke({"question": "Can you remember my name?"})
    print(ai_message)


def with_memory_interaction():
    """
    This function shows an example of what a chat app would look like with memory.
    :return:
    """
    chat_history = [
        HumanMessage("Hi, I'm Henry."),
        AIMessage("Hi Henry, How can I help you today?"),
    ]
    template = (
        "Given the following conversation:"
        "\nChat History:\n"
        "{history}"
        "\n, and a follow-up question:\n"
        "{question}\n"
        "respond to the follow-up question in english:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chat_context = RunnablePassthrough.assign(history=lambda _: chat_history)
    chain = chat_context | prompt | llm | StrOutputParser()
    ai_message = chain.invoke({"question": "Can you remember my name?"})
    print(ai_message)


def simple_chat():
    """
    This function shows an example of a simple chat app using the GoogleGenerativeAI model.
    :return:
    """

    def load_memory(
        return_messages=True, input_key="human_input", memory_key="chat_history"
    ):
        """
        Load memory
        :param return_messages:
        :param input_key:
        :param memory_key:
        :return:
        """
        memory = ConversationBufferMemory(
            return_messages=return_messages, input_key=input_key, memory_key=memory_key
        )
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
            | itemgetter(memory_key)
        )
        return memory, loaded_memory

    template = (
        "you are a chatbot having a conversation with a human, \n"
        "Given the following conversation:"
        "\nChat History:\n"
        "{chat_history}"
        "\n, and a follow-up question:\n"
        "{question}\n" + "respond to the follow-up question in english:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    memory, loaded_memory = load_memory()
    chat_context = RunnablePassthrough.assign(
        chat_history=lambda loaded_memory: get_buffer_string(
            loaded_memory["chat_history"]
        )
    )
    chain = loaded_memory | chat_context | prompt | llm | StrOutputParser()
    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break
        result = chain.invoke({"question": question})
        print(result)
        memory.chat_memory.add_message(HumanMessage(question))
        memory.chat_memory.add_message(AIMessage(result))


def simple_rag():
    """
    This function shows and example of a simple rag app using the GoogleGenerativeAI model.
    :return:
    """

    def get_vector_index():
        """
        This function returns the context for the RAG model.
        :return:
        """
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_document"
        )

        loader = TextLoader("./../data/state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(docs, embeddings_model)
        return vectorstore

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = GoogleGenerativeAI(model="gemini-pro")
    vector_index = get_vector_index()
    retriever = vector_index.as_retriever()

    chat_context = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    )
    chain = chat_context | prompt | llm | StrOutputParser()
    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break
        result = chain.invoke({"question": question})
        print(result)


if __name__ == "__main__":
    simple_rag()
