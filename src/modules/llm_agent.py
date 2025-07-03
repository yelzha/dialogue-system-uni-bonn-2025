# modules/llm_agent.py

from langchain.chains import RetrievalQA
from modules.llm_provider import OllamaLLM

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_question(rag_chain, query: str) -> str:
    return rag_chain.run(query)
