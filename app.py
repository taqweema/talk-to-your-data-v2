import os
import streamlit as st
from dotenv import load_dotenv
from langdetect import detect
import openai
import re

from utils.file_utils import read_file
from utils.rag_engine import process_and_query
from utils.image_utils import extract_text_from_image

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page setup
st.set_page_config(
    page_title="Talk to Your Data V2",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS and JS for citation click handling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: auto;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        background-color: #ffffff;
        color: #333333;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .upload-area h4, .upload-area p {
     color: #333333 !important;
    }
    .stTextInput > div > div > input {
        padding: 1rem;
        font-size: 1rem;
        border: 1px solid #ccc;
        border-radius: 8px;
    }
    .result-block {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .uploadedFileName {
        font-size: 0.95rem;
        color: #333;
        margin-top: 0.75rem;
    }
    .citation-ref {
        color: blue;
        cursor: pointer;
        font-weight: bold;
    }
    </style>
    <script>
    function openSidebarFromCitation(refNum) {
        const sidebar = window.parent.document.querySelector("section[data-testid='stSidebar']");
        if (sidebar) sidebar.style.display = "block";
    }
    window.addEventListener('DOMContentLoaded', () => {
        const citations = document.querySelectorAll(".citation-ref");
        citations.forEach(el => {
            el.addEventListener('click', () => openSidebarFromCitation(el.innerText));
        });
    });
    </script>
""",
    unsafe_allow_html=True,
)

# --- Header Section ---
st.markdown(
    """
    <div class='header'>
        <h1>Talk to Your Data V2</h1>
        <p style='margin-bottom: 2rem;'>Turn your documents into insights — just upload and ask.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- Upload Area Box Info ---
st.markdown(
    """
    <div class='upload-area'>
        <h4>📂 Upload your documents to begin</h4>
        <p style='margin: 0.5rem 0 0 0;'>Limit 200MB per file • PDF, TXT, DOCX, CSV, XLS, PPTX, JPG, JPEG, PNG</p>
    </div>
""",
    unsafe_allow_html=True,
)

# --- File Upload ---
uploaded_files = st.file_uploader(
    label="",
    type=["pdf", "txt", "docx", "csv", "xls", "pptx", "jpg", "jpeg", "png"],
    label_visibility="collapsed",
    accept_multiple_files=True,
    key="main_file_uploader",
)

# --- File Handling ---
file_text = ""
if uploaded_files:
    combined_texts = []

    for uploaded_file in uploaded_files:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            if uploaded_file.type in ["image/jpeg", "image/png"]:
                extracted = extract_text_from_image(uploaded_file)
            else:
                extracted = read_file(uploaded_file)

        if extracted and "[ERROR]" not in extracted:
            combined_texts.append(extracted)
            st.markdown(
                f"<p class='uploadedFileName'>📄 {uploaded_file.name}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"⚠️ Skipped file: {uploaded_file.name} (unsupported or error)")

    file_text = "\n\n".join(combined_texts)

# --- Question Input & Result Display ---
if file_text:
    user_question = st.text_input(
        "Ask a question about your data", placeholder="e.g., What are the key findings?"
    )

    if user_question:
        lang = detect(user_question)
        if lang == "ur":
            prompt = f"سوال: {user_question}\nجواب اردو میں دیں۔"
        elif lang == "en":
            prompt = f"Question: {user_question}\nAnswer in English. Use bullets or numbered format where appropriate."
        else:
            prompt = f"{user_question}\nAnswer in the same script and tone."

        with st.spinner("Thinking..."):
            result = process_and_query(file_text, prompt, return_sources=True)

        answer = result.get("answer")
        usage = result.get("usage", {})
        sources = result.get("sources", {})

        if answer:
            # Convert [1] style tags into clickable spans
            def make_clickable(text):
                return re.sub(
                    r"\[(\d+)\]", r"<span class='citation-ref'>[\1]</span>", text
                )

            answer_with_clicks = make_clickable(answer)
            st.markdown(
                "<div class='result-block'><h4>Results</h4><p>"
                + answer_with_clicks.replace("\n", "<br>")
                + "</p></div>",
                unsafe_allow_html=True,
            )

        if sources:
            st.sidebar.markdown("### 📚 Citations")
            for ref_num, meta in sources.items():
                st.sidebar.markdown(
                    f"""
                **[{ref_num}] Page {meta['page_number']}**
                > {meta['excerpt']}
                """
                )

        if usage:
            total_tokens = getattr(usage, "total_tokens", 0)
            cost = (total_tokens / 1000) * 0.005
            st.markdown(
                f"<p style='text-align:center; color:gray;'>🧾 Used <strong>{total_tokens}</strong> tokens – Approx cost: <strong>${cost:.4f}</strong></p>",
                unsafe_allow_html=True,
            )
