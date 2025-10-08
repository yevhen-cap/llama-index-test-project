from os import getenv, environ
from typing import Union

import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

INDEX_NAME = getenv("INDEX_NAME", "test-index")

environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

llm = AzureOpenAI(
    engine=getenv("MODEL_NAME"),
    model=getenv("MODEL_NAME"),
    temperature=0.0
)


def get_chroma_index_vector_store() -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(f"./db/{INDEX_NAME}")
    chroma_collection = chroma_client.get_or_create_collection("new-collection")
    chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    vector_store: VectorStoreIndex = VectorStoreIndex.from_vector_store(
        vector_store=chroma_vector_store,
        embed_model=AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME"))
    )

    return vector_store


def start_chat(index: VectorStoreIndex):
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            verbose=True,
            llm=llm
        )

    st.set_page_config(
        page_title="Project-X chatting tool",
        page_icon="🤦‍♂️",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("Test project chatting tool 🤦‍♂️")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me about these templates?",
            }
        ]

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(message=prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)


def main() -> None:
    index = None
    logger.info("Running ChromaDB")
    index = get_chroma_index_vector_store()
    start_chat(index)


if __name__ == "__main__":
    main()
