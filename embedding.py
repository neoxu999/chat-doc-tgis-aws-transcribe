import os
import tempfile

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis


class DocEmbedding:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def create_doc_embedding(self, file_path, redis_url, index_name) -> None:
        """
        Stores document embeddings using Langchain and FAISS
        """
        # Create a temporary file and write the PDF data to it
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        #     tmp_file.write(file)
        #     tmp_file_path = tmp_file.name

        # Now you can use the file path with PyPDFium2Loader
        loader = PyPDFium2Loader(file_path)
        documents = loader.load()

        all_splits = self.text_splitter.split_documents(documents)
        print(f"Loaded {len(all_splits)} document chunks.")

        # Store the embeddings vectors using Redis
        rds = Redis.from_documents(all_splits,
                                   self.embeddings,
                                   redis_url=redis_url,
                                   index_name=index_name)
        rds.write_schema("redis_schema.yaml")
        print(f"Stored document to index: {rds.index_name}")
        # os.remove(file_path)

    def get_doc_retriever(self, redis_url, index_name, schema):
        """
        Retrieves document embeddings
        """
        rds = Redis.from_existing_index(
            self.embeddings,
            redis_url=redis_url,
            index_name=index_name,
            schema=schema,
        )
        retriever = rds.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 4, "similarity": 0.3})
        return retriever
