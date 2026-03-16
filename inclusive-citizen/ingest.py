import os
import requests
import urllib3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CHROMA_PATH = "./chroma_db"

DOCUMENTS_TO_INGEST = [
    {
        "url": "https://budidiesel.gov.my/Download/Faq/FAQ_BUDI_Individu.pdf",
        "local_path": "./data/budi_madani.pdf",
        "category": "budi"
    },
    {
        "url": "https://bantuantunai.hasil.gov.my/FAQ/FAQ%20SUMBANGAN%20ASAS%20RAHMAH%20(SARA)%202026.pdf",
        "local_path": "./data/mysara.pdf",
        "category": "mysara"
    }
]

def database():
    os.makedirs("./data", exist_ok=True)
    all_documents = []

    #dowanload PDF
    print("Downloading PDF")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for doc_info in DOCUMENTS_TO_INGEST:
        print (f"Processing: {doc_info['url']}")
        category_tag = doc_info["category"]
    
        try:
            response = requests.get(doc_info['url'], headers=headers, verify=False, timeout=15)

            if response.status_code == 200:
                with open(doc_info['local_path'], "wb") as f:
                    f.write(response.content)
                print("Download successful")

                loader = PyPDFLoader(doc_info['local_path'])
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["category"] = category_tag

                all_documents.extend(docs)

            else:
                print(f"Failed to download PDF. {response.status_code}")
                
        except Exception as e:
            print(f"Connection error: {e}")

    #parse & chunk
    print("Chunking text")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 100,
        length_function = len,
    )
    chunks = text_splitter.split_documents(all_documents)

    #embed & store
    print("Building vector db")
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print("Database build success")

if __name__ == "__main__":
    database()

