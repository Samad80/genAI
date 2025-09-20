from transformers import pipeline
import torch
import streamlit as st

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
st.title("Privacy Policy summarization with TinyLlama")

app_name = st.selectbox("App Name", ["WhatsApp", "Facebook", "Instagram", "Twitter", "Snapchat"])


if st.button("Summarize"):
    with st.spinner("Summarizing..."):
        messages = [
            {
                "role": "privacy policy expert",
                "content": "You are a privacy policy expert. Summarize the privacy policy of {app_name} in a few sentences:"
            },
            {
                "role": "user",
                 "content": f"Summarize the privacy policy of {app_name} in 4-6 concise simple bullet points. Use plain language so that a layperson can quickly understand what data is collected, how it's used, if it's shared with others, and what choices the user has."
            }
            ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        st.write("Summary:")
        st.write(outputs[0]["generated_text"])
            