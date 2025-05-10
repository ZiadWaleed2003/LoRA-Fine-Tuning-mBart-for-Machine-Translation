import streamlit as st
import requests

# Replace with your ngrok server URL
NGROK_API_URL = "https://5bcc-34-41-230-135.ngrok-free.app/api/predict"

st.title("Arabic-English Translator 🌍")


examples = [
    "Select an example...",
    "مرحبا بكم في تطبيق الترجمة",
    "الذكاء الاصطناعي هو مستقبل التكنولوجيا",
    "أريد تعلم اللغة الإنجليزية",
    "كيف يمكنني مساعدتك اليوم؟"
]


selected_example = st.selectbox("Choose a sample Arabic sentence:", examples)

# Input area
initial_text = selected_example if selected_example != "Select an example..." else ""
arabic_text = st.text_area("Enter Arabic Text:", value=initial_text, height=200, placeholder="Type or paste Arabic text here...")

st.markdown("### Instructions:")
st.markdown("1. Enter Arabic text or choose from the dropdown.")
st.markdown("2. Click 'Translate' to get English translation.")

# Translation
if st.button("Translate"):
    if arabic_text.strip():
        try:
            response = requests.post(NGROK_API_URL, json={"text": arabic_text})

            
            if response.status_code == 200:

                response_json = response.json()

                st.text_area("Translated English Text:", response_json, height=200)
            
            else:
                st.error(f"Server error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to the translation server: {e}")
    else:
        st.warning("Please enter Arabic text to translate.")
