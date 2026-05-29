import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA

# 1. Initialize FastAPI (The Web Server)
app = FastAPI()

# 2. Setup Local Embeddings (The Librarian's Eyes)
# We use 'all-MiniLM-L6-v2'. It's a small but powerful model that 
# converts text to math locally on your machine for free.
print("Loading Local Embeddings Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load the Vector Database (The Librarian's Memory)
# This finds the 'chroma_db' folder created by your ingest.py script.
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")

db = Chroma(
    persist_directory=db_path, 
    embedding_function=embeddings
)

# 4. Setup Local LLM (The Librarian's Voice)
# This uses Ollama to run Llama3. Make sure 'ollama run llama3' works in your terminal!
print("Connecting to Ollama (Llama3)...")
llm = Ollama(model="llama3")

# 5. Create the RAG Pipeline (The Brain Logic)
# This chain: 
#   a) Searches the DB for relevant code (retriever)
#   b) Gives that code + your question to Llama3 (llm)
#   c) Returns a natural language answer.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}) # 'k:3' means it looks at the top 3 code snippets
)

# 6. Define the Data Structure for the API
class Question(BaseModel):
    query: str

# 7. Endpoint: Ask a question about your code
@app.post("/ask")
async def ask_question(item: Question):
    print(f"User is asking: {item.query}")
    try:
        # The AI processes the question here
        response = qa_chain.invoke(item.query)
        return {"answer": response["result"]}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": "The AI Engine encountered an error. Is Ollama running?"}

# 8. Endpoint: Health Check
@app.get("/")
def home():
    return {
        "status": "Local AI Engine is Online",
        "model": "Llama3",
        "database": "ChromaDB"
    }