import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_docs():
    # Load Arabic documents
    loader = DirectoryLoader('documents/', glob="**/*.txt")
    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in the 'documents/' folder. Please add .txt files.")

    # Arabic text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۔", "؟", "!", "。", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    if not splits:
        raise ValueError("No text splits generated. Check the document content and splitting logic.")

    # Multilingual embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Create FAISS vector store
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local("faiss_index")
        print("Documents ingested and FAISS index created successfully.")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")

if __name__ == "__main__":
    ingest_docs()