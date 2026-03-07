from langchain_core.prompts import ChatPromptTemplate

query_expansion_prompt = ChatPromptTemplate.from_template(
    "Expand the user query into 1-3 distinct expansions that keep the original intent intact. "
    "Return only the expansions, each on its own line, without numbering or extra text. "
    "Example:\n"
    "Query: What are the health benefits of green tea?\n"
    "Expansions:\n"
    "What are the health benefits of drinking green tea?\n"
    "How does green tea improve health?\n"
    "What positive effects does green tea have on health?\n"
    "\n"
    "Query: {query}"
)
