from dotenv import load_dotenv
import os
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

import os

groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL_NAME")
embedding_model = os.environ.get("EMBEDDING_MODEL_NAME")


# For Windows users, you might need to set the tesseract command path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

import streamlit as st
# PyPDF2 is no longer needed for extraction
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from htmlTemplates import bot_template, user_template, css

# -------------------- PDF & Text Chunking (with OCR) --------------------
pytesseract.pytesseract.tesseract_cmd =r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_pdf_text_with_ocr(pdf_files, languages):
    """
    Extracts text from PDF files using OCR.
    Converts each PDF page to an image and uses Tesseract to extract text.
    """
    text = ""
    # Convert tesseract language codes (e.g., ['eng', 'fra']) to the format needed ('eng+fra')
    lang_str = '+'.join(languages)
    
    for pdf_file in pdf_files:
        # Read PDF from bytes
        images = convert_from_bytes(pdf_file.read())
        for image in images:
            # Use Tesseract to extract text, specifying languages
            content = pytesseract.image_to_string(image, lang=lang_str)
            if content:
                text += content + "\n"  # Add a newline after each page's content
    return text

def get_chunk_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# -------------------- Embedding + Vector Store --------------------

def get_vector_store(text_chunks):
    # This now uses the multilingual model specified in your .env file
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# -------------------- LLM Chain Setup (No changes needed here) --------------------

def get_conversation_chain(vector_store):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL_NAME, temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    system_template = """
    Use the following pieces of context and chat history to answer the question at the end.
    The context can be in any language. Answer the question based on the context.
    If you don't know the answer, just say you don't know, don't make anything up.

    Context: {context}
    Chat history: {chat_history}
    Question: {question}
    Helpful Answer:
    """

    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question", "chat_history"],
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )
    return conversation_chain

# -------------------- Chat Handler (No changes needed here) ----

def handle_user_input(question):
    try:
        response = st.session_state.conversation.invoke({'question': question})
        st.session_state.chat_history = response['chat_history']
        
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# -------------------- UI (Modified for Language Selection) --------------------

def main():
    st.set_page_config(page_title='Chat with PDFs', page_icon='📄')
    st.write(css, unsafe_allow_html=True)
    st.header('📄 Chat with PDFs ')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask anything about your document(s):")

    if question:
        if st.session_state.conversation:
            handle_user_input(question)
        else:
            st.warning("⚠ Please upload and process PDF first.")

    with st.sidebar:
        st.subheader("Your Documents")
        
        # https://tesseract-ocr.github.io/tessdoc/Data-Files-in-version-4.00-information.html
        # Common Tesseract language codes
        available_langs = {
            "English": "eng", "Spanish": "spa", "French": "fra", 
            "German": "deu", "Chinese (Simplified)": "chi_sim", "Japanese": "jpn",
            "Korean": "kor", "Russian": "rus", "Hindi": "hin", "Arabic": "ara"
        }
        
        selected_languages = st.multiselect(
            "Select language(s) in the PDF:",
            options=list(available_langs.keys()),
            default=["English"]
        )
        
        pdf_files = st.file_uploader("Choose PDF(s) & press Process", type=['pdf'], accept_multiple_files=True)

        if pdf_files and st.button("Process"):
            if not selected_languages:
                st.warning("⚠ Please select at least one language.")
                return

            lang_codes = [available_langs[lang] for lang in selected_languages]

            with st.spinner("🔄 Processing PDFs with OCR... This may take a while."):
                try:
                    raw_text = get_pdf_text_with_ocr(pdf_files, lang_codes)
                    if not raw_text.strip():
                        st.error("❌ Could not extract text. The PDF might be empty or corrupted.")
                        return
                    
                    chunks = get_chunk_text(raw_text)
                    vector_store = get_vector_store(chunks)

                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("✅ PDFs processed! You can now ask questions.")
                except Exception as e:
                    st.error(f"❌ Error while processing PDFs: {e}")

if __name__ == "__main__":
    main()
