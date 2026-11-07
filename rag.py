from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

def build_index(extracted_dir="data/extracted"):
    loader = DirectoryLoader(
        extracted_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local("faiss_rbi")
    print("FAISS index built successfully")
    return vs

def rag_answer(question: str, entity_type: str = None) -> dict:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local("faiss_rbi", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve top 5 relevant chunks
    docs = vs.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Use the same small model that works with current transformers
    llm = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )
    
    prompt = f"""Answer the question using only the context below. If the answer is not in the context, say 'Not covered in the documents'.

Context:
{context}

Question: {question}
Answer:"""
    
    result = llm(prompt, return_full_text=False)[0]["generated_text"].strip()
    
    sources = [
        {"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "?")}
        for doc in docs
    ]
    
    return {"answer": result, "sources": sources}