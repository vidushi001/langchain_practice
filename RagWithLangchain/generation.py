from langchain_ollama  import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=1.5)

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def generate_answer(messages):
    output = llm.invoke(messages)
    return output