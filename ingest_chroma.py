import pathlib as pl
from os import getenv, environ

import chromadb
from faker import Faker
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex, Document)
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from llama_index.llms.azure_openai import AzureOpenAI


INDEX_NAME = getenv("INDEX_NAME")

environ["OPENAI_API_VERSION"] = "2024-02-01"

chroma_client = chromadb.PersistentClient(f"./db/{INDEX_NAME}")
faker = Faker()
llm = AzureOpenAI(
    engine=getenv("EMBED_MODEL_NAME"),
    model=getenv("EMBED_MODEL_NAME"),
)

def create_chroma_index() -> None:
    if not pl.Path(f"./db/{INDEX_NAME}").exists():
        logger.info(f"Creating index: {INDEX_NAME}")
        chroma_client.get_or_create_collection("new-collection")


def ingest_documents() -> None:
    logger.info("Read Directory with files")
    reader = SimpleDirectoryReader(
        "./data/data/TEACHER", recursive=True, required_exts=[".pdf"]
    )
    logger.info("Load data")
    tmp_documents = reader.load_data(num_workers=4)

    documents: list[Document] = list()
    for document in tmp_documents:
        name = f"{faker.first_name()} {faker.last_name()}"
        text = f"{name}\n{document.text}"
        documents.append(Document(text=text, metadata={"name": name}))
    np: SimpleNodeParser = SimpleNodeParser.from_defaults(
        chunk_size=512, chunk_overlap=20
    )

    logger.info("Creating nodes")
    np.get_nodes_from_documents(documents)

    Settings.llm = llm
    Settings.embed_model = AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME"))
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    logger.info("Configuring ChromaDB index")
    collection = chroma_client.get_or_create_collection("new-collection")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Creating index from documents")
    VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True
    )


if __name__ == "__main__":
    create_chroma_index()
    ingest_documents()
