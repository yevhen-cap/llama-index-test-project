import asyncio
from os import getenv

from llama_index.core import Settings
from src.chroma_retriever import ChromaRetriever
from llama_index.core.agent.workflow import AgentStream, ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.query_engine import RetrieverQueryEngine

from src.tools.chroma_tools import get_chroma_vector_store


llm = AzureOpenAI(
    engine=getenv("MODEL_NAME"),
    model=getenv("MODEL_NAME"),
    temperature=0.0
)

Settings.llm = llm

retriever = ChromaRetriever(
    vector_store=get_chroma_vector_store(),
    embed_model=AzureOpenAIEmbedding(model=getenv("EMBED_MODEL_NAME")),
    query_mode="default",
    similarity_top_k=2
)

query_engine = RetrieverQueryEngine.from_args(retriever)


def query_retrieval_tool(query: str):
    '''retrieve info from chromadb with retriever'''

    return query_engine.query(query)


def query_general_tool(prompt: str):
    '''Answer general questions'''
    response = llm.complete(prompt)
    return response.text

retrieval_tool = FunctionTool.from_defaults(
    fn=query_retrieval_tool,
    name="retrieval_tool",
    description="Use this tool to retrieve information from the ChromaDB using semantic search.",
)

general_tool = FunctionTool.from_defaults(
    fn=query_general_tool,
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
    prompt = input("Ask your question(type exit for exit): ")
    if prompt.strip().lower() == "exit":
        print("Bye!")
        exit(0)

    handler = agent.run(prompt)
    async for ev in handler.stream_events():
        # if isinstance(ev, ToolCallResult):
        #     print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)
    await handler

if __name__ == "__main__":
    while True:
        asyncio.run(main())