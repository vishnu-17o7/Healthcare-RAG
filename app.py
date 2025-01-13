import os
import textwrap
import streamlit as st
import requests
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llama_parse_key = os.getenv("LLAMA_PARSE")
api_key = os.getenv("WEATHER_API")

# Load PDF and process documents
def read_pdf(file_path):
    import fitz
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Read medical document
pdf_text = read_pdf("Breast_Cancer.pdf")
document_path = Path("data/parsed_document.md")
with document_path.open("w", encoding="utf-8") as f:
    f.write(pdf_text)

# Weather API call
def get_weather(location):
    url1 = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response1 = requests.get(url1)
    if response1.status_code == 200:
        weather = response1.json()
        temperature = weather['current']['temp_c']
        return temperature
    else:
        st.error(f"Failed to retrieve weather data: {response1.status_code}")
        return None

# Load documents into Qdrant for RAG
def load_documents():
    loader = UnstructuredMarkdownLoader(document_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    docs = text_splitter.split_documents(loaded_documents)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        path="./db",
        collection_name="medical_embeddings",
    )
    return qdrant

# Set up recommendation prompt with dynamic user input
recommendation_prompt = """
Use the following information to answer the user's question.

Context: {context}
"Details:" : [
Heart Rate: {{heart_rate}}
Blood Pressure: {{blood_pressure}}
Steps Today: {{steps_today}}
Temperature: {{temperature}}
User Query: {{user_query}}

]
Answer the question in JSON format.
Example:
{{
  "recommendations": [
    {{
      "type": "exercise",
      "advice": "Increase daily steps to 7000 for better cardiovascular health."
    }},
    {{
      "type": "diet",
      "advice": "Consume more potassium-rich foods to maintain healthy blood pressure."
    }}
  ]
}}
"""
def print_response(response):
    # Handle the result text
    response_txt = response.get("result", "")
    
    # Print result text
    for chunk in response_txt.split("\n"):
        if not chunk:
            st.write("")  # Adds a blank line
            continue
        # Wrap the text and display it using st.write
        st.write("\n".join(textwrap.wrap(chunk, 100, break_long_words=False)))
    
    # Handle the recommendations if they exist in the response
    recommendations = response.get("recommendations", [])
    if recommendations:
        st.write("\n**Health Recommendations:**")
        for recommendation in recommendations:
            st.write(f"- **Type:** {recommendation['type']}")
            st.write(f"- **Advice:** {recommendation['advice']}")
            st.write("")  # Add a blank line for better readability

# Initialize the LLM and RAG pipeline
def get_retrieval_qa(qdrant):
    prompt = PromptTemplate(
        template=recommendation_prompt, 
        input_variables=["context", "user_query", "heart_rate", "blood_pressure", "steps_today", "temperature"]
    )

    llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=qdrant.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": False},
    )
    return qa

# Streamlit UI
def main():
    st.title("Health Recommendation System")
    
    # User Query Input
    user_query = st.text_input("Ask a health-related question:")

    # Input patient data
    heart_rate = st.number_input("Enter Heart Rate", min_value=40, max_value=200, value=85)
    blood_pressure = st.text_input("Enter Blood Pressure (e.g., 120/80)", value="120/80")
    steps_today = st.number_input("Enter Steps Today", min_value=0, value=4500)
    location = st.text_input("Enter Location for Weather", value="Coimbatore")

    # Check if the Qdrant is already created
    if "qdrant" not in st.session_state:
        # Load documents into Qdrant only once
        qdrant = load_documents()
        st.session_state.qdrant = qdrant  # Save the Qdrant object in session state

    # Load Qdrant from session state
    qdrant = st.session_state.qdrant

    if st.button("Generate Health Recommendations"):
        # Get weather for the location
        temperature = get_weather(location)
        if temperature is None:
            return  # Stop if weather data is unavailable

        # Get the QA system
        qa = get_retrieval_qa(qdrant)

        # Execute the RAG pipeline
        response = qa.invoke({
            "query": "Generate health recommendations based on the patient's data.",
            "user_query": user_query,
            "heart_rate": heart_rate,
            "blood_pressure": blood_pressure,
            "steps_today": steps_today,
            "temperature": temperature,
        })

        # Print the result
        st.write("Health Recommendations:")
        print_response(response)

# Run the Streamlit app
if __name__ == "__main__":
    main()
