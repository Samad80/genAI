import gradio as gr
import os, shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline


CHROMA_PATH = "/content/chroma"

# ---------------- Models ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 300, "temperature": 0.2},
)

ANSWER_PROMPT = """
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

EXPLANATION_PROMPT = """
Explain WHY the following context was relevant to answering the question.

Question:
{question}

Context:
{context}

Give a short explanation.
"""

db = None  # vector DB (created after upload)

# ---------------- Functions ----------------
def index_pdf(pdf_file):
    global db

    if pdf_file is None:
        return "❌ No PDF uploaded."

    # Clear old DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Gradio gives a FILE PATH (new versions)
    pdf_path = pdf_file

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Create DB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    db.persist()

    return f"✅ Indexed PDF successfully ({len(chunks)} chunks)"


def ask_question(question):
    global db

    if db is None:
        return "⚠️ Please upload and index a PDF first.", "", ""

    results = db.similarity_search_with_relevance_scores(question, k=3)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    answer = llm.invoke(
        ChatPromptTemplate.from_template(ANSWER_PROMPT).format(
            context=context, question=question
        )
    )

    explanation = llm.invoke(
        ChatPromptTemplate.from_template(EXPLANATION_PROMPT).format(
            context=context, question=question
        )
    )

    sources = []
    for doc, score in results:
        page = doc.metadata.get("page", "N/A")
        source = os.path.basename(doc.metadata.get("source", "PDF"))
        sources.append(f"{source} — page {page}, relevance {score:.2f}")

    return answer, explanation, "\n".join(sources)

# ---------------- UI ----------------
with gr.Blocks() as ui:
    gr.Markdown("## 📘 Explainable PDF Question Answering")

    pdf_input = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath",   # ⭐ THIS IS CRITICAL
    )

    index_btn = gr.Button("📥 Index PDF")
    status = gr.Textbox(label="Status")

    index_btn.click(
        fn=index_pdf,
        inputs=pdf_input,
        outputs=status,
    )

    question = gr.Textbox(label="Ask a question")
    answer = gr.Textbox(label="📘 Answer")
    explanation = gr.Textbox(label="🧠 Why this answer?")
    sources = gr.Textbox(label="📄 Sources")

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[answer, explanation, sources],
    )

ui.launch(share=True)
