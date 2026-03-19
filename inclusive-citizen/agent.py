import json
import re
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

#connect to local ollama & db
llm = Ollama(model="llama3", temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#DB1: policy database
gov_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
gov_retriever = gov_db.as_retriever(search_kwargs={"k": 2})

#DB2: dialect database
dialect_db = Chroma(persist_directory="./dialect_db", embedding_function=embeddings)
dialect_retriever = dialect_db.as_retriever(search_kwargs={"k": 3})

#define prompt
detect_prompt = PromptTemplate.from_template("""
Anda adalah pakar linguistik. Analisis teks pengguna ini.
Gunakan CONTOH TERJEMAHAN di bawah sebagai panduan untuk memahami loghat tersebut.

CONTOH TERJEMAHAN YANG DITEMUI:
{dialect_context}

TUGASAN:
1. Kenal pasti loghat yang digunakan.
2. Terjemah teks tersebut ke dalam Bahasa Melayu Standard yang rasmi.

PENTING: Jawab HANYA dalam format JSON yang sah dengan kunci "detected_dialect" dan "standard_query".
TEKS PENGGUNA: {user_input}
""")

master_answer_prompt = PromptTemplate.from_template("""
You are a highly empathetic, friendly Malaysian public servant chatting with a citizen on WhatsApp. 
Your job is to read the complex government policy below and explain it in extremely simple terms.

CRITICAL RULES YOU MUST FOLLOW (FAILING TO DO SO IS A CRITICAL ERROR):
1. VOCABULARY MIRRORING (CASUAL MALAY): Reply in highly casual, spoken Malay (Bahasa Melayu Pasar). You MUST reuse the exact slang words the user typed in their question to build empathy. Look at this dictionary context to understand what they meant, and reuse their slang:
   {dialect_context}
2. NO ENGLISH & NO NAMES: Do NOT use English words. NEVER treat the slang words as human names.
3. ZERO JARGON (5TH GRADE LEVEL): DO NOT copy-paste long sentences. You are strictly forbidden from using acronyms like "SKDS", "KPDN", or "T20". Replace them with simple words like "Kerajaan" (Government) or "orang kaya" (rich people).
4. HUMAN TONE: DO NOT call the user "pengguna" or sound like a robot. NEVER say "Saya akan menjawab soalan anda". 
5. TOPIC FOCUS: Do not mix up government programs. If they ask about MySARA, only talk about MySARA.
6. ACTION-ORIENTED: If they ask "How to apply?", immediately give the official website link or steps.
7. ANSWER DIRECTLY: You must explicitly answer all parts of the user's question.
8. NO HALLUCINATION: If the information is not in the Official Context, say exactly: "Maaf, maklumat ini tiada dalam rujukan rasmi saya."
9. NO META-COMMENTARY (CRITICAL): NEVER explain your thought process. DO NOT add notes at the end like "(Note: I used the exact slang...)". Output the template and immediately STOP.

OFFICIAL CONTEXT:
{gov_context}

USER QUESTION: {user_input}

MANDATORY OUTPUT TEMPLATE (YOU MUST FILL THIS IN EXACTLY IN MALAY/DIALECT, KEEP THE ASTERISKS):
[Friendly greeting using dialect/casual Malay]

* [Bullet Point 1: Direct, simple answer to the question without jargon]
* [Bullet Point 2: Important condition or requirement from the text]
* [Bullet Point 3: The exact website link or clear action to take]
""")

#processing function
def process_citizen_query(user_input):
    try:
        #detect dialect
        print(f"\nNEW QUERY: {user_input}")
        print("Searching dialect database for context")
        dialect_docs = dialect_retriever.invoke(user_input)
        dialect_context = "\n-".join([d.page_content for d in dialect_docs])
        print(f"Dialect context found:\n{dialect_context}")
        raw_detect = llm.invoke(detect_prompt.format(
            dialect_context = dialect_context,
            user_input = user_input
        ))

        try:
            clean_text = raw_detect.replace("```json","").replace("```","").strip()
            match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
            if match:
                clean_json = match.group(0)
                routing_data = json.loads(clean_json)
                dialect = routing_data.get("detected_dialect", "Bahasa Melayu Santai")
                standard_query = routing_data.get("standard_query", user_input)
            else:
                raise ValueError("No JSON found in LLM response.")
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            dialect = "Bahasa Melayu Santai"
            standard_query = user_input

        print("Analyzing query for document routing")
        user_input_lower = user_input.lower()
        search_filter = None

        if "mysara" in user_input_lower or "sara" in user_input_lower:
            search_filter = {"category": "mysara"}
            print(" [Router]: Keyword 'mysara' detected.")
        elif "budi madani" in user_input_lower or "budi" in user_input_lower or "diesel" in user_input_lower:
            search_filter = {"category": "budi"}
            print(" [Router]: Keyword 'budi madani' or 'budi' detected.")

        print("Searching government database")
        if search_filter:
            gov_docs = gov_db.similarity_search(standard_query, k=6, filter=search_filter)
        else:
            gov_docs = gov_db.similarity_search(standard_query, k=6)

        gov_context = "\n".join([d.page_content for d in gov_docs])

        final_answer = llm.invoke(master_answer_prompt.format(
            dialect = dialect,
            dialect_context = dialect_context,
            gov_context = gov_context,
            user_input = user_input
        ))

        return final_answer, dialect, standard_query

    except Exception as e:
        return f"System Error: {str(e)}", "Unknown", user_input