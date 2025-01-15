from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from dotenv import load_dotenv
from os.path import join, dirname
import logging


class Rag:
    '''
    Wrapper around the langchain RAG implementation.
    '''

    def __init__(self, connection: str, embeddings: str, llm: str, llm_type: str):
        embeddings = HuggingFaceEmbeddings(model_name = embeddings, model_kwargs = {})
        db = PGVector(connection, embeddings)
        llm = CTransformers(model = llm,
                            model_type = lllm_type,
                            max_new_tokens = 256,
                            temperature = 0.5)
        prompt = PromptTemplate("Question:{query} Context:{context}", input_variables = ['query', 'context'])
        self._chain = RetrievalQA.from_chain_type(llm = llm,
                                                  retriever = db.as_retriever(search_kwargs={'k':3}),
                                                  chain_type = "stuff",
                                                  chain_type_kwargs={'prompt': prompt})

    def respond(self, prompt: str):
        return self._chain({'query': prompt})['result']


if __name__ == "__main__":
    load_dotenv(join(dirname(__file__), "rag.env"))
    env = os.environ
    connection = f"postgresql+psycopg://{env.get('PG_USER')}:{env.get('PG_PASS')}@{env.get('PG_HOST')}:5432/{env.get('PG_DB')}"

    rag = Rag(connection, env.get('EMBEDDINGS_MODEL'), env.get('TRANSFORMER_MODEL'), env.get('TRANSFORMER_MODEL_TYPE'))

    while True:
        resp = rag.respond(input("Query:"))
        print("Response:\n" + resp)