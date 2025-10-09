import asyncio
from os import environ, getenv

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
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


def query_retrieval_tool(query: str):
    '''retrieve info from chromadb'''
    index = get_chroma_index_vector_store()
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)

    return "\n\n".join([n.get_content() for n in nodes]) 


def query_general_tool(prompt: str):
    '''Answer general questions'''
    response = llm.complete(prompt)
    return response.text


# Wrap retrieval tool
def retrieval_tool_fn(query: str) -> str:
    """fhsdkjjfdls"""
    return query_retrieval_tool(query)

retrieval_tool = FunctionTool.from_defaults(
    fn=retrieval_tool_fn,
    name="retrieval_tool",
    description="Use this tool to retrieve information from the ChromaDB using semantic search.",
)

# Wrap general tool
def general_tool_fn(prompt: str) -> str:
    """general tool"""
    return query_general_tool(prompt)

general_tool = FunctionTool.from_defaults(
    fn=general_tool_fn,
    name="general_tool",
    description="Use this tool to answer general questions using Azure OpenAI.",
)   


# Create ReAct agent
agent: ReActAgent = ReActAgent(
    tools=[retrieval_tool, general_tool],
    llm=llm,
    verbose=True,
)

ctx = Context(agent)


def react_agent(user_input: str) -> str:
    return agent.chat(user_input)


async def main() -> None:
    logger.info("Running ChromaDB")
    
    prompt = input("Ask your question: ")

    handler = agent.run(prompt)
    async for ev in handler.stream_events():
        # if isinstance(ev, ToolCallResult):
        #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)
    response = await handler

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nBye!")
            exit(0)
