from langchain_core.prompts import ChatPromptTemplate

rerank_pointwise_prompt = ChatPromptTemplate.from_template(
    "You are a retrieval reranker.\n"
    "Score how relevant the passage is for answering the query.\n"
    "Return only a single integer between 1 and 10.\n"
    "1 means irrelevant. 10 means highly relevant.\n"
    "Query: {query}\n"
    "Passage: {chunk}\n"
    "Score:"
)
