# Python RAG wrapper around langchain functionality for local execution.

* Install PostgreSQL (16.x) https://www.postgresql.org
* Build and install pgvector https://github.com/pgvector/pgvector
* Configure environment variables in rag.env
* Store the embeddings from source code in the vector db: python vector_db.py
* Run the transformer: python rag.py