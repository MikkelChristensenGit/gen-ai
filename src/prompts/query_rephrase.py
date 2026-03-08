from langchain_core.prompts import ChatPromptTemplate

query_rephrase_prompt = ChatPromptTemplate.from_template(
    "You are a query rewriter for retrieval.\n"
    "Rewrite the user's message into one clear, standalone search query for semantic retrieval.\n"
    "\n"
    "Rules:\n"
    "- Preserve the original intent.\n"
    "- Keep key entities, dates, versions, and constraints.\n"
    "- Resolve ambiguous references when possible from the message itself.\n"
    "- Do not answer the question.\n"
    "- Do not add new facts.\n"
    "- Return exactly one line: only the rewritten query.\n"
    "\n"
    "User message: {query}\n"
    "Rewritten query:"
)
