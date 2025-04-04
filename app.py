
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Kiswahili Grammar Tutor", page_icon="🧠")

# Load model and tokenizer from Hugging Face Hub
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Bur3hani/kiswmod")
    model = AutoModelForSeq2SeqLM.from_pretrained("Bur3hani/kiswmod")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

st.title("🧠 Kiswahili Grammar Tutor")
st.markdown("An AI assistant that acts like a Kiswahili teacher or therapist. It corrects sentences and explains grammar in a conversational way (max 3 turns).")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Mwanafunzi:", "")

if st.button("Submit") and user_input:
    # Build dialogue history
    dialogue = ""
    for student, teacher in st.session_state.history:
        dialogue += f"Mwanafunzi: {student}\nMwalimu: {teacher}\n"
    dialogue += f"Mwanafunzi: {user_input}\nMwalimu:"

    # Generate response
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
    output_ids = model.generate(**inputs, max_length=128)
    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.session_state.history.append((user_input, reply))
    if len(st.session_state.history) > 3:
        st.session_state.history = st.session_state.history[-3:]

# Display conversation
for student, teacher in st.session_state.history:
    st.markdown(f"**Mwanafunzi:** {student}")
    st.markdown(f"**Mwalimu:** {teacher}")
