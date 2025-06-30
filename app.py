# app.py
import streamlit as st
from chatbot import chatbot
from speech_recognizer import transcribe_speech

st.set_page_config(page_title="Voice-Enabled Chatbot", page_icon="💬")

st.title("🗣️ Voice-Enabled Chatbot")
st.markdown("Ask a question by typing or speaking. The chatbot will respond based on the story context.")

input_mode = st.radio("Choose input mode:", ("Text", "Speech"))

user_input = ""

if input_mode == "Text":
    user_input = st.text_input("💬 Enter your question:")

elif input_mode == "Speech":
    if st.button("🎤 Start Recording"):
        with st.spinner("Listening..."):
            user_input = transcribe_speech()
        st.write(f"📝 Transcribed text: `{user_input}`")

# Display chatbot response
if user_input:
    with st.spinner("Generating response..."):
        response = chatbot(user_input)
    st.markdown(f"**🤖 Chatbot:** {response}")
