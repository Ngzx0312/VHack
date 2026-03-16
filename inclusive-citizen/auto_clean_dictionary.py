import os
import json
import re
from datasets import load_dataset
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#configuration
DIALECT_DB_PATH = "./dialect_db"
TWEETS_TO_PROCESS = 500

llm = Ollama(model="llama3", temperature=0.1)

#prompt
extraction_prompt = PromptTemplate.from_template("""
You are a data extraction bot. Look at the Malaysian dialect sentence and its Standard Malay translation below.
Extract 1 to 3 key dialect/slang words and their Standard Malay meanings.

Respond ONLY with a valid JSON array of objects. Do not include any other text, greetings, or markdown.
Format example: [{{"loghat": "pitih", "standard": "wang"}}, {{"loghat": "guano", "standard": "bagaimana"}}]

DIALECT SENTENCE: {dialect_sentence}
STANDARD TRANSLATION: {standard_sentence}

JSON OUTPUT:
""")

def build_dictionary():
    print("Connecting to Hugging Face")
    dataset = load_dataset(
       "mesolitica/chatgpt4-noisy-translation-twitter-dialect", 
        split="train", 
        streaming=True 
    )

    clean_dictionary = []
    count = 0

    print ("Processing tweets and building dictionary")

    for row in dataset:
        if count >= TWEETS_TO_PROCESS:
            break

        r_data = row.get("r", {})
        original_data = row.get("original", {})

        dialect_text = r_data.get("translation", "")
        standard_text = original_data.get("ms", "")

        if dialect_text and standard_text:
            print(f"\nAnaliyzing Tweet {count + 1}")
            print(f"Messy Text: {dialect_text[:50]}...")

            #extraction
            raw_extraction = llm.invoke(extraction_prompt.format(
                dialect_sentence = dialect_text,
                standard_sentence = standard_text
            ))

            #clean output and parse the JSON
            try:
                clean_text = raw_extraction.replace("```json", "").replace("```", "").strip()
                match = re.search(r'\[.*?\]', clean_text, re.DOTALL)

                if match:
                    word_pairs = json.loads(match.group(0))
                    for pair in word_pairs:
                        if "loghat" in pair and "standard" in pair:
                            clean_dictionary.append(pair)
                            print(f"Extracted: {pair['loghat']} = {pair['standard']}")
            except Exception as e:
                print("AI fialed to format this tweet. Skip")

            count += 1

    print("Converting extracted words into Vector Documents")
    documents = []
    for item in clean_dictionary:
        text = f"Kamus Loghat: '{item['loghat'].lower()}' bermaksud '{item['standard'].lower()}'."
        documents.append(Document(page_content=text))

    print("Building clean Vector DB")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    db = Chroma.from_documents(documents, embeddings, persist_directory=DIALECT_DB_PATH)
    print(f"\nDictionary built with {len(clean_dictionary)} entries and saved to {DIALECT_DB_PATH}")

if __name__ == "__main__":
    build_dictionary()