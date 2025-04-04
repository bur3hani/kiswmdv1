import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

st.set_page_config(page_title="Kiswahili Grammar Tutor", page_icon="🧠")

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Bur3hani/kiswmod")
    model = AutoModelForSeq2SeqLM.from_pretrained("Bur3hani/kiswmod")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("🧠 Kiswahili Grammar Tutor")
st.markdown(
    """
    A respectful and knowledgeable assistant that acts like a Kiswahili **teacher or therapist**.  
    It can help you with grammar, proper usage, and understanding — in a conversational way.  
    *Mwalimu will guide you up to 3 exchanges per topic.*
    """)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Mwanafunzi:", "")

def clean_response(text):
    text = re.sub(r"<extra_id_\\d+>", "", text)
    text = text.replace("<pad>", "").strip()
    return text

if st.button("Submit") and user_input:
    # Build conversational prompt
    dialogue = ""
    for student, teacher in st.session_state.history:
        dialogue += f"Mwanafunzi: {student}\nMwalimu: {teacher}\n"
    dialogue += f"Mwanafunzi: {user_input}\nMwalimu:"

    # Generate model response
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
    outputs = model.generate(**inputs, max_length=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(response)

    # Update chat history
    st.session_state.history.append((user_input, response))
    if len(st.session_state.history) > 3:
        st.session_state.history = []

# Display chat
for student, teacher in st.session_state.history:
    st.markdown(f"**🧑 Mwanafunzi:** {student}")
    st.markdown(f"**👩🏾‍🏫 Mwalimu:** {teacher}")
