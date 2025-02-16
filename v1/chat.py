from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


def load_model():
    # Arabic-optimized model
    model_name = "aubmindlab/aragpt2-base"  # Smaller, faster Arabic GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
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


def main():
    # Load components
    llm = load_model()

    # Initialize embeddings with keyword arguments
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load FAISS vector store (with deserialization enabled)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Arabic chat interface
    print("مرحبا! أسألني أي سؤال عن مستنداتك. اكتب 'خروج' للإنهاء")
    while True:
        query = input("\nالسؤال: ")
        if query.lower() == "خروج":
            break

        # Use .invoke() instead of __call__
        result = qa_chain.invoke({"query": query})
        print(f"\nالجواب: {result['result']}")
        print("\nالمصادر:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}")


if __name__ == "__main__":
    main()