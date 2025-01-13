# Health Advisory System with RAG

A sophisticated health recommendation system that combines user health metrics with Retrieval-Augmented Generation (RAG) to provide personalized health advice and recommendations.

## Features

- **Health Metrics Monitoring**
  - Heart rate tracking
  - Blood pressure monitoring
  - Daily steps counter
  - Temperature tracking

- **Interactive Query System**
  - Natural language health-related questions
  - Personalized responses based on user metrics
  - Structured health recommendations

- **Advanced RAG Implementation**
  - Multiple LLM integrations:
    - Local LLM using Ollama (llama3)
    - Cloud LLM using Groq (llama-3.1-70b-versatile)
  - Document processing with LangChain
  - Multiple vector store options:
    - ChromaDB for web content
    - Qdrant for medical data
  - Advanced text processing:
    - Efficient text splitting with RecursiveCharacterTextSplitter
    - PDF document parsing using PyMuPDF
    - Contextual compression with FlashrankRerank
  - High-quality embeddings using BAAI/bge-base-en-v1.5

## Technology Stack

- **Frontend & UI**
  - Streamlit
  - Text wrapping for better readability
  - Interactive input forms
  - JSON-formatted responses

- **Backend & Processing**
  - LangChain for document processing and chains
  - Multiple LLM providers:
    - Ollama for local processing
    - Groq for cloud processing
  - Vector Stores:
    - ChromaDB for web content
    - Qdrant for medical data
  - FastEmbed embeddings with BAAI/bge-base-en-v1.5
  - PyMuPDF for PDF processing
  - Weather API integration for environmental context

- **Data Processing**
  - RecursiveCharacterTextSplitter for optimal chunking
  - FlashrankRerank for context compression
  - Structured JSON output format

## Prerequisites

- Python 3.x
- Ollama with llama3 model installed
- Required Python packages:
  ```
  streamlit
  langchain
  langchain_community
  ollama
  chromadb
  qdrant-client
  PyMuPDF
  python-dotenv
  requests
  fastembed
  langchain-groq
  ```
- API Keys (store in .env file):
  ```
  GROQ_API_KEY=your_groq_api_key
  WEATHER_API=your_weather_api_key
  ```

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is installed and the llama3 model is available

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Enter your health metrics:
   - Heart rate (bpm)
   - Blood pressure
   - Steps taken today
   - Body temperature

3. Ask health-related questions and receive personalized recommendations

## Features in Detail

### Health Metrics Input
- Real-time input validation
- Comprehensive health parameter tracking
- User-friendly interface for data entry

### Advanced RAG System
- Multiple document sources support (PDF, web content)
- Efficient document retrieval with dual vector store approach
- Context-aware responses with FlashrankRerank
- Integration with both local and cloud LLMs
- Customizable chunk sizes for optimal performance
- Environmental context integration (weather data)

### Recommendation Engine
- Structured health advice
- Context-based suggestions
- Multiple recommendation categories:
  - Exercise recommendations
  - Health advisories
  - General wellness tips

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is designed to provide general health recommendations and should not be used as a substitute for professional medical advice. Always consult with healthcare professionals for medical decisions.