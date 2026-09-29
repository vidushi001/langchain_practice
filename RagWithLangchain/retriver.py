from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing Chroma database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="documents",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
