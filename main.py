from os import getenv, environ

import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from chromadb import Collection

INDEX_NAME = getenv("INDEX_NAME", "test-index")

environ["OPENAI_API_VERSION"] = "2024-12-01-preview"

llm = AzureOpenAI(
    engine=getenv("MODEL_NAME"),
    model=getenv("MODEL_NAME"),
    temperature=0.0
)

embed_model = AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME"))

Settings.llm = llm
Settings.embed_model = embed_model


def get_full_prompt(name: str) -> str:
    prompt = (f"generate me brief summary about {name} CV."
              "it must contain next information in format:"
              "{Full name}\n"
              "{Profession}\n"
              "{years of experience}\n"
              "{strongest skills}\n"
              "{professional highlights}")
    
    return prompt

@st.cache_resource
def get_chroma_collection() -> Collection:
    chroma_client = chromadb.PersistentClient(f"./db/{INDEX_NAME}")
    chroma_collection = chroma_client.get_or_create_collection("new-collection")

    return chroma_collection

@st.cache_resource
def get_chroma_index_vector_store() -> VectorStoreIndex:
    chroma_vector_store = ChromaVectorStore(chroma_collection=get_chroma_collection())

    vector_store: VectorStoreIndex = VectorStoreIndex.from_vector_store(
        vector_store=chroma_vector_store,
        embed_model=AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME")),      
    )

    return vector_store

@st.cache_resource
def get_names_from_documents():
    result = get_chroma_collection().get(include=["metadatas"])

    names = set([mt["name"] for mt in result["metadatas"]])
    return names



def start_chat():
    search_name = ""
    st.set_page_config(
        page_title="Project-X chatting tool",
        page_icon="🤦‍♂️",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    for item in get_names_from_documents():
        if st.sidebar.button(item):
            search_name = item

    st.title("Test project chatting tool 🤦‍♂️")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Click on these names and get info about experience",
            }
        ]

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    elif search_name != "":
        st.session_state.messages.append({"role": "user", "content": search_name})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_engine = get_chroma_index_vector_store().as_chat_engine()
                response = chat_engine.chat(get_full_prompt(search_name))
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)


def main() -> None:
    logger.info("Running ChromaDB")
    start_chat()



if __name__ == "__main__":
    main()
