import os
import typing
from pathlib import Path
from typing import Iterator

import weaviate
from openpyxl.packaging.manifest import mimetypes
from unstructured.partition.auto import partition
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import google.generativeai as genai
import diskcache as dc
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain import hub
from langchain_core.runnables import RunnableParallel
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())

# Create a cache object
cache = dc.Cache(".cache")

system_instruction = """
 you are an automotive assistant tasked with summarizing files for retrieval. \
 These summaries will be embedded and used to retrieve the original file. \
 Describe concisely the characteristics and teh content of the files. \
 may sure to only describe the content.
 """


def cache_results(func):
    """
    Decorator to cache results of a function.
    """

    def wrapper(*args, **kwargs):
        """
        Wrapper function to cache results.
        """
        # Create a cache key based on the function name and arguments
        # ignore self argument for instance methods
        if len(args) > 0 and hasattr(args[0], "__class__"):
            cache_key = f"{func.__name__}_{args[1:]}_{kwargs}"
        else:
            cache_key = f"{func.__name__}_{args}_{kwargs}"

        if cache_key in cache:
            print(f"Loading cached results for {cache_key}")
            return cache[cache_key]
        else:
            print(
                f"No cache found for {cache_key}. Computing results and saving to cache."
            )
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result

    return wrapper


class GeminiSummarizationLoader(BaseLoader):

    def __init__(self, model: str, docs: list = None):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.docs = docs
        self.model = genai.GenerativeModel(model, system_instruction=system_instruction)

    @cache_results
    def _process_pdf(self, file: str):
        """
        Process a PDF file.
        """
        file = Path(file)
        images_folder = file.parent / file.stem
        images_folder.mkdir(exist_ok=True)
        elements = partition(
            str(file),
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],  # optional
            # can be used to convert them into base64 format
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=str(images_folder),
        )
        return elements

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load documents.
        """
        for doc in self.docs:
            doc = Path(doc)
            mime_type = mimetypes.guess_type(str(doc))[0]
            if "pdf" in mime_type:
                elements = self._process_pdf(str(doc))
                for element in elements:
                    if element.category == "Image":
                        image_path = element.metadata.image_path
                        image_summary = f"""
                        This is the description of the image at the page number {element.metadata.page_number},
                        file: {doc}, image: {image_path}:
                        {self.summarize_image(image_path)} 
                        """
                        yield Document(
                            page_content=image_summary,
                            metadata={
                                "file": str(doc),
                                "image": str(element.metadata.image_path),
                                "page_number": element.metadata.page_number,
                                "category": "Image",
                            },
                        )
                    else:
                        yield Document(
                            page_content=element.text,
                            metadata={
                                "file": str(doc),
                                "image": "",
                                "page_number": element.metadata.page_number,
                                "category": element.category,
                            },
                        )

    @cache_results
    def summarize_image(self, image: str):
        """
        Summarize an image.
        """
        image = Image.open(image)
        prompt = ["Summarize the following image", image]
        response = self.model.generate_content(prompt)
        return response.text


class SummarizationRag:
    """
    A RAG model for summarization.
    """

    def __init__(
        self,
        chat_model,
        embeddings_model,
        summarization_model: str,
        docs: list = None,
        temperature=0.0,
        chunk_size=700,
        chunk_overlap=100,
        collection_name="rag",
    ):
        self.chat_model = ChatGoogleGenerativeAI(
            model=chat_model, temperature=temperature
        )
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model=embeddings_model, task_type="retrieval_document"
        )
        self.summarization_model = summarization_model
        self.docs = docs
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.weaviate_client = weaviate.connect_to_embedded()
        self.weaviate_vector_store = WeaviateVectorStore(
            client=self.weaviate_client,
            index_name=self.collection_name,
            embedding=self.embeddings_model,
            text_key="page_content",
        )

    def add_file(self, file: str):
        """
        Add a file to the vector index.
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File {file} not found")

        file_ext = file.suffix
        if file_ext not in [".pdf", ".mp3", ".wav", ".mp4", ".jpg", ".jpeg", ".png"]:
            raise ValueError(f"File type {file_ext} not supported")
        self.docs.append(file)

    def populate_vector_index(self):
        """
        This function returns the vector index.
        :return:
        """
        try:
            self.weaviate_client.connect()
            docs = self.get_docs()
            batch_docs_size = 50
            batch_docs = [
                docs[i : i + batch_docs_size]
                for i in range(0, len(docs), batch_docs_size)
            ]

            for batch_idx, docs in enumerate(batch_docs):
                self.weaviate_vector_store.add_documents(docs)
                print(f"Indexed {len(docs)} documents in batch {batch_idx + 1}")
        finally:
            self.weaviate_client.close()

    def reset_vector_index(self):
        """
        Reset the vector index.
        """

        try:
            self.weaviate_client.connect()
            collection = self.weaviate_client.collections.get(self.collection_name)
            if collection:
                self.weaviate_client.collections.delete(self.collection_name)
        finally:
            self.weaviate_client.close()

    def get_docs_count_at_vector_index(self):
        """
        Get the number of documents in the vector index.
        """
        self.weaviate_client.connect()
        try:
            collection = self.weaviate_client.collections.get(self.collection_name)
            count = 0
            for _ in collection.iterator():
                count += 1
                if count > 100000:
                    break
        finally:
            self.weaviate_client.close()
        return count

    def get_docs(self) -> typing.List[Document]:
        """
        Summarize an audio file.
        """
        loader = GeminiSummarizationLoader(self.summarization_model, self.docs)
        docs = loader.load()
        return docs

    def get_rag_chain(self):
        """
        Creates a Retrieval-Augmented Generation QA chain.
        """

        def format_docs(docs):
            """
            Format the documents.
            """
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = hub.pull("rlm/rag-prompt")
        retriever = self.weaviate_vector_store.as_retriever()
        llm = self.chat_model
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )
        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)
        return rag_chain_with_source

    def get_rag_chain_with_chat_history(self):
        """
        Creates a Retrieval-Augmented Generation QA chain with chat history.
        """

        retriever = self.weaviate_vector_store.as_retriever()
        llm = self.chat_model

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain_with_chat_history = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        return rag_chain_with_chat_history

    def chat(self):
        """
        Main function to execute the RAG workflow.
        """

        # Creating RAG QA Chain
        print(
            "Starting chat session..., num indexed docs: ",
            self.get_docs_count_at_vector_index(),
        )

        try:
            self.weaviate_client.connect()
            qa_chain = self.get_rag_chain_with_chat_history()
            console = Console()
            chat_history = []

            while True:
                question = input("Question: ")
                if question.lower() == "exit":
                    break
                # result = qa_chain.invoke(question)
                result = qa_chain.invoke(
                    {"input": question, "chat_history": chat_history}
                )
                chat_history.extend([HumanMessage(content=question), result["answer"]])
                print(f"Answer: {result['answer']}")
                docs = result["context"]
                for doc in docs[:3]:
                    console.print(
                        f"Doc: {doc.metadata['file']}, Page: {doc.metadata['page_number']}"
                    )
                    console.print(f"Content: {doc.page_content}")
                    console.print()
        finally:
            self.weaviate_client.close()


if __name__ == "__main__":
    # Create RAGSystem instance and execute main function
    rag = SummarizationRag(
        summarization_model="models/gemini-1.5-flash-latest",
        chat_model="models/gemini-1.5-flash-latest",
        embeddings_model="models/embedding-001",
        docs=["./../data/papers/vis-language-model.pdf"],
    )

    # cache.clear()
    # rag.reset_vector_index()
    # rag.populate_vector_index()
    rag.chat()
