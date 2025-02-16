from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)

SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"]


def load_documents():
    loaders = {
        ".txt": UnstructuredFileLoader,
        ".pdf": UnstructuredPDFLoader,
        ".docx": UnstructuredWordDocumentLoader
    }

    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        try:
            loader = DirectoryLoader(
                'documents/',
                glob=f"**/*{ext}",
                loader_cls=loaders[ext],
                show_progress=True
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    if not documents:
        raise ValueError("No supported documents found. Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS))

    return documents
