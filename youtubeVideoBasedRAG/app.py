from generation  import generate_answer;
from retriever import retriever
from augmentation import augmentation

query = 'what is langchain and give its component as well'
docs = retriever.invoke(query)

messages = augmentation(query,docs)
# Ask Ollama
response = generate_answer(messages)
print(response.content)