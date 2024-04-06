# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Install required libraries
# # !pip install -r requirements.txt
# -

# Import required libraries
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


# +
# Define a function to generate response

def generate_response(uploaded_files, api_key, question):
    # Process each uploaded file and store it in a list
    documents = []
    for file in uploaded_files:
        # Read and decode each file and store in the list
        documents.append(file.read().decode())
    
    # Proceed to split, embed and retrieve
    
    # Splitting documents into chunks of manageable size
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splitter.create_documents(documents)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Store embeddings in a vector DB
    vector_db = Chroma.from_documents(docs, embeddings)
    # Create retriever interface
    retriever = vector_db.as_retriever()
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api_key), chain_type='stuff', retriever=retriever)
    
    return qa.run(question)


# +
# Stream UI

# Page title and description
st.set_page_config(page_title='Ask the Document')
st.title('🔖Ask the Document')

st.header('About the app')
st.write('''
This app is an advanced Q&A platform that allows users to upload multiple text documents and receive answers to their queries
based on the content of these documents. The app utilizes RAG framework powered by OpenAI GPT model to provide insightful and
contectually relevant answers.

#### How it works:

1. Upload a document: You can upload any text document in .txt or .pdf format.
2. Ask a Question: Type in your question pertaining to uploaded documents.
3. Get an Answer: The app analyzes the documents and provides relevant answers from the documents.

##### Get Started
Simple upload your documents and start asking questions!!
''')


# File uploader
uploaded_files = st.file_uploader('Upload documents', type='txt', accept_multiple_files=True) 

# Input question
question = st.text_input('Enter question', placeholder='Please provide a short summary', disabled = not uploaded_files)


# Form to take API key and get response
result=[]
with st.form('myform', clear_on_submit=True):
    # get api key
    api_key = st.text_input('Enter OpenAI API key:', type='password', disabled = not(uploaded_files and question))
    # click submit button
    submit = st.form_submit_button('Submit', disabled = not(uploaded_files and question))
    # check if submit button is clicked and openai api key is valid
    if (submit and api_key.startswith('sk-')):
        with st.spinner('Processing...'):
            response = generate_response(uploaded_files, api_key, question)
            # appending response to a list that allows storing the response that can be queried later by the user
            result.append(response) 
            del api_key
            
    # display the reponse
    if len(result):
        st.info(response)
        



