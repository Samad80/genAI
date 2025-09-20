from transformers import pipeline
import torch
import streamlit as st

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
st.title(" Your friendly AI Therapist")
user_input = st.text_input("You: ", "give a prompt to the model")
outputs = [{"generated_text": ""}]
if st.button("Generate"):
    if user_input:
        messages = [
            {
                "role": "Therapist",
                "content": "You are a friendly therapist who always responds kindly and gives the best advice and recommendations.",
        },
        {"role": "user", "content": user_input}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        if outputs:
            st.text_area("Therapist: ", value=outputs[0]["generated_text"], height=200)
