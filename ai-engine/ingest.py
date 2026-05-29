import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Switched from OpenAI
from langchain_community.vectorstores import Chroma

def process_codebase():
    script_dir = Path(__file__).parent
    target_dir = (script_dir / "../frontend/src").resolve()
    
    print(f"Checking folder: {target_dir}")
    
    all_docs = []
    extensions = ['.js', '.jsx', '.ts', '.tsx', '.css']
    for ext in extensions:
        for file_path in target_dir.rglob(f"*{ext}"):
            print(f"Loading: {file_path.name}")
            loader = TextLoader(str(file_path), encoding='utf-8')
            all_docs.extend(loader.load())

    print(f"--- Found {len(all_docs)} files total ---")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)

    # USE LOCAL EMBEDDINGS (FREE)
    print("Using local HuggingFace model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Creating vectors for {len(docs)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=str(script_dir / "chroma_db")
    )
    print("Success! Local Brain is ready.")

if __name__ == "__main__":
    process_codebase()