from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

all_chunks=[]

loader = PyPDFLoader("../document/Job-Offer-Vidushi.pdf")
documents = loader.load()
print(len(documents))


chunks = text_splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

for i,chunk in enumerate(chunks):
    all_chunks.append({
        "id" : len(all_chunks),
        "source" : "Job-Offer-Vidushi.pdf",
        "text" : chunk
    })


for chk in all_chunks:
    print("chunk details here",chk['id'])

# 3. Create embeddings using Sentence Transformer
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store chunks + embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="documents"
)

print("Documents stored in Chroma successfully!")    

    