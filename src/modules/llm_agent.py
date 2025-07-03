# modules/llm_agent.py

from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from modules.llm_provider import OllamaLLM
from modules.agent_tools import tools

def get_tool_agent() -> AgentExecutor:
    """
    Initializes a ReAct-based agent with access to data analysis tools.
    """
    llm = OllamaLLM()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


def get_rag_chain(vectorstore):
    """
    Initializes a simple RetrievalQA chain using a retriever + Ollama LLM.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def answer_question(rag_chain, query: str) -> str:
    """
    Simple interface for answering user queries using RAG chain.
    """
    try:
        return rag_chain.run(query)
    except Exception as e:
        return f"[Error] Failed to answer question: {e}"
