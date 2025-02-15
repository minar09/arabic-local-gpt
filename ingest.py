import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def ingest_docs():
    # Load Arabic documents
    loader = DirectoryLoader('documents/', glob="**/*.txt")
    documents = loader.load()

    # Arabic text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۔", "؟", "!", "。", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    # Arabic embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="aubmindlab/bert-base-arabertv02",
        model_kwargs={'device': 'cpu'}
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    ingest_docs()
    