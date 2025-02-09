from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Defining LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Initialize embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the path where the FAISS index will be saved
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from the CSV file using CSVLoader
    loader = CSVLoader(file_path="All-2479-Answers-retrieved-from-MedQuAD.csv")
    data = loader.load()

    # Create a FAISS instance for the vector database from the loaded documents
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save the vector database locally for later use
    vectordb.save_local(vectordb_file_path)
    print(f"Vector database created and saved to {vectordb_file_path}")

def get_qa_chain():
    # Load the vector database from the local path with dangerous deserialization allowed
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings=instructor_embeddings,
        allow_dangerous_deserialization=True
        )

    # Create a retriever from the vector database with a threshold for scoring
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define a custom prompt template for the QA chain    
    prompt_template = """
        You are a knowledgeable medical care advisor. Use the information provided to answer the user's question accurately.
        If the answer is not in the context, respond with 'I don't know.'

        CONTEXT: {context}
        QUESTION: {question}
    """


    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain using the retriever and the custom prompt template
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain
