from os import getenv

import chromadb
import streamlit as st
from chromadb import Collection
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


@st.cache_resource
def get_chroma_collection() -> Collection:
    index_name = getenv("INDEX_NAME", "test-index")
    chroma_client = chromadb.PersistentClient(f"./db/{index_name}")
    chroma_collection = chroma_client.get_or_create_collection("new-collection")

    return chroma_collection


@st.cache_resource
def get_chroma_vector_store() -> VectorStoreIndex:
    chroma_vector_store = ChromaVectorStore(chroma_collection=get_chroma_collection())

    return chroma_vector_store


@st.cache_resource
def get_chroma_index_vector_store() -> VectorStoreIndex:
    chroma_vector_store = get_chroma_vector_store()

    chroma_index_vector_store: VectorStoreIndex = VectorStoreIndex.from_vector_store(
        vector_store=chroma_vector_store,
        embed_model=AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME")),      
    )

    return chroma_index_vector_store


@st.cache_resource
def get_names_from_documents():
    result = get_chroma_collection().get(include=["metadatas"])

    names = set([mt["name"] for mt in result["metadatas"]])
    return names
