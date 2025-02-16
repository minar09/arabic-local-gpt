import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from utils import load_documents  # Add this import


SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"]


def create_hybrid_retriever(splits):
    # Semantic Retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Keyword Retriever
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    # Save BM25 retriever
    with open("bm25_retriever.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

def ingest_docs():
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۔", "؟", "!", "。", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    retriever = create_hybrid_retriever(splits)
    return retriever


if __name__ == "__main__":
    ingest_docs()
    print("Documents processed and hybrid index created successfully.")