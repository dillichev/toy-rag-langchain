from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from os.path import join, dirname
import logging
import os


class VectorDbLoader:
    '''
    Loads documents from a directory and stores them in a PostgreSQL database as vectors.
    '''

    def __init__(self, connection: str, model: str, directory: str, mask: str):
        self._conn_string = connection
        self._model = model
        self._directory = directory
        self._mask = mask


    def load(self):
        loader = DirectoryLoader(self._directory, self._mask)
        splitter = RecursiveCharacterTextSplitter(chunk_size = 512)

        logging.info("Loading/splitting documents...")
        docs = splitter.split_documents(loader.load())

        logging.info("Getting embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name = self._model, model_kwargs = {})

        logging.info("Storing vectors into database...")
        db = PGVector(embeddings = embeddings, collection_name = "vectors", connection = self._conn_string)
        db.add_documents(docs)
        logging.info("Done")


if __name__ == "__main__":
    load_dotenv(join(dirname(__file__), "rag.env"))
    env = os.environ
    connection = f"postgresql+psycopg://{env.get('PG_USER')}:{env.get('PG_PASS')}@{env.get('PG_HOST')}:5432/{env.get('PG_DB')}"

    loader = VectorDbLoader(connection, env.get('EMBEDDINGS_MODEL'), env.get('SOURCE_PATH'), env.get('SOURCE_PATTERN'))
    loader.load()
