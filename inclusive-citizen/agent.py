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
Anda ialah seorang pegawai perkhidmatan awam Malaysia yang sangat mesra, empati, dan sedia membantu.
Gunakan HANYA MAKLUMAT RASMI di bawah untuk menjawab soalan.

SYARAT-SYARAT KETAT:
1. PENCERMINAN KOSA KATA (BAHASA MELAYU PASAR): Balas dalam Bahasa Melayu Pasar yang sangat santai. Anda WAJIB menggunakan semula perkataan loghat atau slanga yang ditaip oleh pengguna dalam soalan mereka untuk membina empati. Rujuk konteks kamus di bawah untuk memahami maksud mereka, dan gunakan semula loghat tersebut dalam jawapan anda:
   {dialect_context}
2. TAHAP BACAAN DARJAH 5 (ANTI-JARGON): DILARANG SAMA SEKALI 'copy-paste' ayat panjang dari rujukan! Terangkan semula dalam bahasa harian yang sangat mudah. JANGAN guna singkatan rumit seperti SKDS, KPDN, atau T20. Ganti dengan perkataan mudah seperti "Kerajaan", "orang kaya", atau "subsidi minyak".
3. DILARANG SAMA SEKALI menggunakan perkataan Inggeris.
4. JANGAN menganggap perkataan loghat sebagai nama orang.
5. NADA MANUSIA: JANGAN panggil pengguna dengan gelaran robotik seperti "pengguna". Gunakan sapaan mesra seperti "Tuan/Puan", "Awak", atau sapaan loghat yang bersesuaian. Berbual secara terus seperti mesej WhatsApp.
6. FOKUS TOPIK: Jangan campur adukkan program bantuan. Jika ditanya mengenai MySARA, jawab MySARA sahaja berdasarkan maklumat.
7. Berikan jawapan ringkas dan mudah difahami (tahap bacaan darjah 5) dalam bentuk 3 langkah (bullet points).
8. Jika maklumat tiada dalam rujukan, jujur cakap "Maaf, maklumat ini tiada dalam rujukan rasmi saya." JANGAN reka jawapan.
9. TERUS KEPADA TINDAKAN: Jika saya tanya "Macam mana nak mohon?", berikan laman web rasmi atau langkah memohon secara terus. Jangan buang masa terangkan definisi dasar.
10. PASTIKAN anda menjawab SEMUA bahagian soalan pengguna secara terus (Contoh: Jika pengguna tanya perlu mohon atau tidak, jawab "Ya" atau "Tidak" dengan jelas).

MAKLUMAT RASMI: 
{gov_context}

SOALAN: {user_input}
JAWAPAN MESRA:
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