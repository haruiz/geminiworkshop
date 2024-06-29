import os
from operator import itemgetter

from docx import Document
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from rich.console import Console

load_dotenv(find_dotenv())


class RAGApp:
    """
    This class defines a simple APP for a RAG (Retrieval-Augmented Generation) system.
    """

    def __init__(
        self,
        chat_model,
        embeddings_model,
        document_path,
        temperature=1,
        chunk_size=700,
        chunk_overlap=100,
    ):
        self.document_path = document_path
        self.chat_model = chat_model
        self.embeddings_model = embeddings_model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_document(self):
        """
        Loads a document and splits it into pages.
        """
        _, file_extension = os.path.splitext(self.document_path)
        if file_extension.lower() == ".pdf":
            pdf_loader = PyPDFLoader(self.document_path)
            return pdf_loader.load_and_split()
        elif file_extension.lower() == ".docx":
            doc = Document(self.document_path)
            pages = [paragraph.text for paragraph in doc.paragraphs]
            return pages
        elif file_extension.lower() == ".txt":
            with open(self.document_path, "r") as file:
                return file.read().split("\n\n")
        else:
            raise ValueError(
                "Unsupported file format. Only PDF, DOCX, and TXT files are supported."
            )

    def get_vector_index(self):
        """
        This function returns the vector index.
        :return:
        """
        pages = self.load_and_split_document()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        context = "\n\n".join(pages)
        chunks = text_splitter.split_text(context)
        vector_index = Chroma.from_texts(chunks, self.embeddings_model)
        return vector_index

    def get_rag_chain(self):
        """
        Creates a Retrieval-Augmented Generation QA chain.
        """

        template = (
            "Answer the question based only on the following context: \n"
            "{context}"
            "\nQuestion: {question}"
        )
        prompt = ChatPromptTemplate.from_template(template)
        llm = self.chat_model
        retriever = self.get_vector_index().as_retriever()

        chat_context = RunnablePassthrough.assign(
            context=itemgetter("question") | retriever
        )
        chain = chat_context | prompt | llm | StrOutputParser()
        return chain

    def chat(self):
        """
        Main function to execute the RAG workflow.
        """

        # Creating RAG QA Chain
        qa_chain = self.get_rag_chain()
        console = Console()
        while True:
            question = input("Question: ")
            if question.lower() == "exit":
                break
            result = qa_chain.invoke({"question": question})
            console.print(result)

    def chat_ui(self):
        """
        Main function to execute the RAG workflow with a simple UI.
        """

        # Creating RAG QA Chain
        qa_chain = self.get_rag_chain()
        import gradio as gr

        def new_message_handler(message, history):
            """
            This function handles the new message.
            :param message:
            :param history:
            :return:
            """
            result = qa_chain.invoke({"question": message["text"]})
            return result

        demo = gr.ChatInterface(
            fn=new_message_handler,
            examples=[{"text": "what questions do you suggest about your context?"}],
            title="Echo Bot",
            multimodal=True,
        )
        demo.launch()


if __name__ == "__main__":
    # Constants
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DOCUMENT_PATH = "./../data/state_of_the_union.txt"

    # Create RAGSystem instance and execute main function
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", task_type="retrieval_document"
    )

    rag_system = RAGApp(
        document_path=DOCUMENT_PATH,
        chat_model=chat_model,
        embeddings_model=embeddings_model,
    )
    rag_system.chat_ui()
