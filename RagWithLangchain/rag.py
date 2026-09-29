from generation  import generate_answer;
from retriver import retriever
from augmentation import augmentation



query = 'Share the monthly salary'
docs = retriever.invoke(query)

messages = augmentation(query,docs)
# Ask Ollama
response = generate_answer(messages)

print(response.content)