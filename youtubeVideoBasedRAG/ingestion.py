from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

url = "https://www.youtube.com/watch?v=VekumySN8os"

loader = YoutubeLoader.from_youtube_url(
    url,
    add_video_info=False
)


documents = loader.load()

print(f"Transcript loaded: {len(documents)} documents")

print(documents[0].page_content[:1000])

chunks = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="documents"
)