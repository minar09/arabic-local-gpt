# Arabic Local LLM Chat

A simple yet powerful solution for chatting with Arabic documents using fully local Large Language Models (LLMs). This repository allows you to ingest Arabic documents, create embeddings, and interact with them through a chat interface. All processing happens locally, ensuring privacy and security.

---

## Table of Contents
1. [Repo Structure](#repo-structure)
2. [Setup Instructions](#setup-instructions)
3. [Key Features](#key-features)
4. [Recommended Models](#recommended-models)
5. [Optimization Tips](#optimization-tips)
6. [Future Enhancements](#future-enhancements)

---

## Repo Structure
arabic-local-gpt/
├── requirements.txt # Python dependencies
├── ingest.py # Script to process and embed documents
├── chat.py # Chat interface to interact with documents
├── models/ # Directory for storing models (optional)
└── documents/ # Place your Arabic documents here


---

## Setup Instructions

### 1. Install Requirements
Install the necessary Python packages using the following command:
```bash
pip install -r requirements.txt
```

### 2. Add Documents
Place your Arabic text files (`.txt`) in the `documents/` folder.

Supported formats: `.txt`, `.pdf` (add `unstructured[pdf]` to `requirements.txt` for PDF support).

### 3. Ingest Documents
Run the ingestion script to process and embed your documents:
```bash
python ingest.py
```

This will create a faiss_index directory containing the vector store.

4. Start Chatting
Launch the chat interface to start interacting with your documents:

```bash
python chat.py
```

You can now ask questions in Arabic, and the system will provide answers based on the ingested documents.

## Key Features
* Fully Local : No external APIs or cloud services required.
* Arabic-Optimized : Uses AraBERT for embeddings and AraGPT2 for text generation.
* Document Sources : Displays the source documents for each answer, ensuring transparency.
* Privacy-Focused : All processing happens locally, keeping your data secure.

