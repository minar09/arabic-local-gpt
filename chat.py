from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr
import pickle


def load_model():
    model_name = "aubmindlab/aragpt2-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        # load_in_4bit=True  # Add 4-bit quantization for gpu
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=pipe)


def create_qa_chain():
    llm = load_model()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load FAISS
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Load BM25 retriever
    with open("bm25_retriever.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever()],
        weights=[0.4, 0.6]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        return_source_documents=True
    )


def gradio_interface(query, history):
    qa_chain = create_qa_chain()
    result = qa_chain.invoke({"query": query})

    sources = "\n".join([f"- {doc.metadata['source']}" for doc in result['source_documents']])
    return f"{result['result']}\n\nالمصادر:\n{sources}"


if __name__ == "__main__":
    # Launch Gradio UI
    gr.ChatInterface(
        gradio_interface,
        title="محادثة مع المستندات العربية",
        description="اسأل أي سؤال عن مستنداتك العربية المحلية",
        examples=["ما هو الموضوع الرئيسي؟", "لخص المحتوى"],
        css="footer {visibility: hidden}"
    ).launch(server_name="0.0.0.0", share=False)
