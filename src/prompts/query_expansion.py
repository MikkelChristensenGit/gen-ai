from langchain_core.prompts import ChatPromptTemplate

query_expansion_prompt = ChatPromptTemplate.from_template(
    "Expand the user query into up to {max_expansions} distinct expansions that keep the original intent intact. "
    "Return only the expansions, each on its own line, without numbering or extra text.\n"
    "Query: {query}"
)
