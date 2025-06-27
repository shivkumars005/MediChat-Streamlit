from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw data from PDF(s) file
DATAPATH = "data/"
def load_pdf_data(file_path):
    loader = DirectoryLoader(file_path,
                             glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_data(DATAPATH)
#print("Length of documents:", len(documents))


# Step 2: Create chunks from text
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50,)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

chunks = create_chunks(documents)
#print("Number of chunks created:", len(chunks))


# Step 3: Create vector embeddings for each chunk
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = get_embedding_model()

# Step 4: Store embeddings in a vector database (FAISS) 
FAISS_DB_PATH = "vector_db/faiss_db"
db = FAISS.from_documents(chunks, embeddings)
db.save_local(FAISS_DB_PATH)

