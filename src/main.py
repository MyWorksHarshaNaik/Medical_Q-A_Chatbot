# main.py
import os
import streamlit as st
from helper import create_vector_db, get_qa_chain

def main():
    # Create the vector database if it doesn't already exist
    if not os.path.exists("faiss_index"):
        create_vector_db()

    # Initialize the QA chain
    chain = get_qa_chain()
    
    # Streamlit app interface
    st.title("Medical Chat Bot")
    
    query = st.text_input("Enter your query:", "Causes of Diabetes?")
    
    if st.button("Get Response"):
        # Use the invoke method instead of __call__
        response = chain.invoke({"query": query})
        
        # Display the response in Streamlit
        st.subheader("Response:")
        st.write(response['result'])

        # st.subheader("Source Documents:")
        # for doc in response['source_documents']:
        #     st.write(f"- {doc.page_content}")

if __name__ == "__main__":
    main()
