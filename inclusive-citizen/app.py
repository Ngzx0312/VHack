import streamlit as st
from agent import process_citizen_query

#configure
st.set_page_config(page_title="Portal Bantuan Rakyat", page_icon=":speech_balloon:", layout="centered")


with st.sidebar:
    st.title("Maklumat Sistem")
    st.info("Sistem ini dikuasakan oleh kecerdasan buatan (AI) secara lokal untuk memastikan privasi")
    st.divider()
    st.markdown("**Pangkalan Data Aktif:**")
    st.markdown("- Bantuan BUDI Madani")
    st.markdown("- Bantuan MySARA")
    st.divider()
    st.caption("Dibina untuk rakyat, oleh rakyat")

#chat interface
st.markdown("""
<style>
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("Portal Bantuan Rakyat")
st.markdown("Selamat datang. Sila tanya apa sahaja soalan mengenai inisiatif bantuan kerajaan, dan kami akan cuba membantu anda dalam bahasa yang paling mudah.")

#initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hai! Ada apa-apa soalan mengenai bantuan kerajaan yang boleh saya bantu?"}
    ]

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#handle user input
if prompt := st.chat_input("Tulis pertanyaan anda di sini..."):
    
    #display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #process
    with st.chat_message("assistant"):
        with st.spinner ("Sedang menyemak dokumen rasmi..."):
            final_answer, _, _ = process_citizen_query(prompt)

            #show final points
            st.markdown(final_answer)

    #save response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})