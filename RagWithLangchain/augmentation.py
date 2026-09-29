from langchain_core.prompts import ChatPromptTemplate

# RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}
""")

def augmentation(query,docs):
    context = "\n\n".join(
    doc.page_content for doc in docs
    )

    # Create prompt
    messages = prompt.invoke({
        "context": context,
        "query": query
    })
    return messages