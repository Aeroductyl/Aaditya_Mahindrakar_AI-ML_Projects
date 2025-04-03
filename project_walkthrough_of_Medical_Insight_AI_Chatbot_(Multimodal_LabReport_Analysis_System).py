
# This is dummy architechure of the project procss to walk through the algorithm.
# Install necessary libraries/
#Many will ne impoted in the bash
pip install langchain openai pypdf pymupdf pandas numpy \
fastapi uvicorn psycopg2-binary sqlalchemy


# SECTION 1
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text


# SECTION 2
# Created a postgreSQL database
# Connecting the Database with python

from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = "postgresql://username:password@localhost/medical_insights"

Base = declarative_base()

class MedicalReport(Base):
    __tablename__ = "medical_reports"
    
    id = Column(Integer, primary_key=True)
    patient_name = Column(Text)
    report_text = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


# SECTION 3
# Initialize database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(engine)

# Function to insert a report
def save_report(patient_name, text):
    session = SessionLocal()
    report = MedicalReport(patient_name=patient_name, report_text=text)
    session.add(report)
    session.commit()
    session.close()


# SECTION 4
# Using Langchain Framework
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.chains import SQLDatabaseChain
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://username:password@localhost/medical_insights"
engine = create_engine(DATABASE_URL)

db = SQLDatabase(engine)
llm = OpenAI(model_name="gpt-4")

# Query medical reports using LangChain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
query = "SELECT report_text FROM medical_reports WHERE patient_name='John Doe';"
print(db_chain.run(query))


# Creating Embeddings for External knowledge input
#from sentence_transformers import SentenceTransformer
#import faiss
#import numpy as np
#import chromadb

# Load pre-trained embedding model
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate vector embeddings
#def generate_embedding(text):
#    return embedding_model.encode(text, convert_to_numpy=True)

# Example usage
#report_text = "Patient has high blood pressure and irregular heart rate."
#vector = generate_embedding(report_text)
#print(vector.shape)  # Output: (384,) for MiniLM


# SECTION 5
#Vector Database for External knowledge input
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS and load stored vectors
faiss_db = FAISS.load_local("faiss_index", embedding_model)

# Search for similar reports
query = "Hypertension and high cholesterol treatment"
similar_reports = faiss_db.similarity_search(query, k=3)
for doc in similar_reports:
    print(doc.page_content)


# SECTION 6
# Experimenting

from langchain.tools import PubMedSearchTool

pubmed_tool = PubMedSearchTool()
research_papers = pubmed_tool.run("Diabetes latest treatment")
print(research_papers)



# SECTION 7
# Creating RAG pipeline

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4")

retriever = FAISS.load_local("faiss_index", embedding_model)
rag_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

query = "What are the latest treatments for hypertension?"
insights = rag_chain.run(query)
print(insights)


#Phase 2: Adding Multimodal Support (X-ray/Scan Analysis) with LangChain
#Now, we'll integrate image analysis into the system to process X-rays, MRI scans, and surgical images. This will allow the AI to analyze both textual reports and medical images for better insights.

#1. Approach for Multimodal RAG
# Extract text from reports (Phase 1 - already built)
# Analyze images (X-ray, MRI, surgical scans) using CNN & Vision Transformer (ViT)
# Store image embeddings in FAISS for retrieval
# Use LangChain to combine text + image insights
# Query both text and image data to generate medical insights

