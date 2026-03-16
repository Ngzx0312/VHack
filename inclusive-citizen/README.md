# Portal Bantuan Rakyat

An advanced, locally hosted AI assistant designed to bridge the digital divide for Malaysian citizens. This system uses a Multi-RAG architecture and Metadata Filtering to understand regional dialects, simplify bureaucratic jargon, and route citizens to correct government policies with zero hallucinations.

## Case Study
Digital government portals often use complex official terminology and a single dominant language. This creates a barrier for elderly users, rural communities, and those with lower digital literacy, leading to exclusion from essential public services.

## Solution & Technical Innovations
An NLP-powered assistant that does not just translate, but simplifies and summarizes complex official information into local dialects.

* **Dialect-Aware Translation (LLM-Powered ETL):** A custom ingestion pipeline that uses a local AI to extract slang from noisy open-source datasets (Mesolitica) and build a clean, secondary Vector Database for real-time dialect translation (e.g., Kelantanese, Sabahan).
* **Zero-Hallucination Routing (Metadata Filtering):** To solve the critical issue of "Cross-Document Contamination", the retriever dynamically applies metadata tags to official PDFs. If a user asks about *MySARA*, the AI is physically walled off from the *BUDI Madani* or *eBelia* documents.
* **Lexical Simplification:** The system strictly translates complex policy documents into a 5th-grade reading level using simple, actionable bullet points.
* **Privacy-First Local Deployment:** Runs 100% offline using Ollama and lightweight models, ensuring sensitive citizen queries are never sent to external cloud APIs.
* **Inclusivity UI:** A clean, accessible chat interface designed for users with low digital literacy.

## Tech Stack
* **Frontend:** Streamlit
* **LLM Engine:** Ollama (Llama 3)
* **Orchestration & RAG:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Hugging Face (paraphrase-multilingual-MiniLM-L12-v2)

## Project Structure
* `app.py` - The Streamlit frontend and user interface.
* `agent.py` - The core AI logic, dialect routing, and metadata filtering.
* `ingest.py` - The script to build the official government policy database.
* `auto_clean_dictionary.py` - The AI-powered ETL script that builds the dialect database.
* `data/` - Directory containing the official PDF documents.
* `requirements.txt` - Python dependencies.

## Quickstart Guide

### Prerequisites
1.  Python 3.9 or higher.
2.  Ollama installed and running on your local machine.

### Installation & Setup

1.  **Clone the repository:**
    git clone [https://github.com/Ngzx0312/VHack.git]
    cd VHack/inclusive-citizen

2.  **Install dependencies:**
    pip install -r requirements.txt

3.  **Pull the local LLM:**
    ollama pull llama3

4.  **Build the Databases:**
    python ingest.py
    python auto_clean_dictionary.py

5.  **Run the Application:**
    python -m streamlit run app.py

## Usage Example
Test the system's dialect and routing capabilities by typing a regional query into the chat:
* "Guano nak mintak pitih bantuan diesel ni?" (Tests Kelantanese dialect + BUDI Madani routing)
* "Siapa je layak dapat mysara, mcm mana nak check?" (Tests standard informal Malay + MySARA routing)
