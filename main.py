import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Enable CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Sawan Portfolio AI Backend Running"}


# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------

loader1 = TextLoader("data/portfolio_data.txt")
loader2 = TextLoader("data/projects.txt")

documents = loader1.load() + loader2.load()

# -----------------------------
# SPLIT TEXT INTO CHUNKS
# -----------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

docs = splitter.split_documents(documents)

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# CREATE VECTOR DATABASE
# -----------------------------

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# GEMINI MODEL
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# -----------------------------
# REQUEST MODEL
# -----------------------------


class Question(BaseModel):
    query: str


# -----------------------------
# CHAT ENDPOINT
# -----------------------------


@app.post("/chat")
def chat(q: Question):

    # Retrieve relevant documents
    results = retriever.invoke(q.query)

    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
You are the AI assistant for Sawan Kumar's portfolio website.

You are an AI assistant for Sawan Kumar's portfolio website.

Answer questions using the provided portfolio knowledge.

You can answer questions about:
- Sawan's skills
- projects
- education
- certifications
- technologies
- hobbies and interests
- languages
- personal background
- portfolio website
- resume
- contact information

If the question is not related to Sawan Kumar, politely say that you only answer questions about Sawan's portfolio.
Important instructions:

If someone asks about Sawan's resume, tell them that they can download it by clicking the "Download Resume" button on the portfolio website.

Also provide the direct resume link as an alternative.

Always mention the button first before showing the link.

If the answer is not in the provided context, say:
"I do not have that information in Sawan's portfolio please contact him for more info. "

Context:
{context}

Question:
{q.query}

Answer:
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}
