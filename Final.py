try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from datetime import datetime


# Convert uploaded file to markdown text
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Reset ChromaDB collection
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# Add text chunks to ChromaDB
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}

    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection

    collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()

        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )

    return collection


# Q&A function
def get_answer(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not docs or min(distances) > 1.5:
        return "Unfortunately, I don't have information about that topic in my holistic library."

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
        {context}

        Question: {question}

        Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with \"I don't know.\" Do not add information from outside the context.

        Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    # return response[0]['generated_text'].strip()
    return str(distances)

# --- Holistic, Calm, Warm Custom CSS ---
def add_holistic_css():
    st.markdown('''
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/6203748/pexels-photo-6203748.jpeg?_gl=1*1t0xtpi*_ga*MTgyNjIzNDgzMy4xNzUxMTQyMTY1*_ga_8JE65Q40S6*czE3NTExNDIxNjQkbzEkZzEkdDE3NTExNDQ1MjUkajI0JGwwJGgw");
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }
    .main .block-container {
        background: rgba(255, 255, 240, 0.88) !important;
        border-radius: 18px !important;
        padding: 2rem 2.5rem 2rem 2.5rem !important;
        box-shadow: 0 4px 32px 0 rgba(0,0,0,0.10) !important;
    }
    .stButton>button {
        background-color: #f7c873 !important;
        color: #5a3e1b !important;
        border-radius: 12px !important;
        border: none !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.6em 1.5em !important;
        margin: 0.2em 0.2em !important;
        box-shadow: 0 2px 8px 0 rgba(87, 60, 13, 0.10) !important;
        transition: background 0.2s !important;
    }
    .stButton>button:hover {
        background-color: #ffe2b0 !important;
        color: #7a4c1e !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #7a4c1e !important;
    }
    .stMetric {
        background: rgba(255, 255, 240, 0.7) !important;
        border-radius: 10px !important;
    }
    </style>
    ''', unsafe_allow_html=True)


# --- Robust file conversion with progress and error handling ---
def safe_convert_files(uploaded_files):
    converted_docs = []
    errors = []
    if not uploaded_files:
        return converted_docs, ["No files uploaded"]
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Converting {uploaded_file.name}...")
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                errors.append(f"{uploaded_file.name}: File too large (max 10MB)")
                continue
            allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in allowed_extensions:
                errors.append(f"{uploaded_file.name}: Unsupported file type")
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                markdown_content = convert_to_markdown(tmp_path)
                if len(markdown_content.strip()) < 10:
                    errors.append(f"{uploaded_file.name}: File appears to be empty or corrupted")
                    continue
                converted_docs.append({
                    'filename': uploaded_file.name,
                    'content': markdown_content,
                    'size': len(uploaded_file.getvalue()),
                    'word_count': len(markdown_content.split())
                })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")
        progress_bar.progress((i + 1) / len(uploaded_files))
    status_text.text("Conversion complete!")
    return converted_docs, errors


# --- Show conversion results ---
def show_conversion_results(converted_docs, errors):
    if converted_docs:
        st.success(f"🌼 Successfully converted {len(converted_docs)} document(s)!")
        total_words = sum(doc['word_count'] for doc in converted_docs)
        st.info(f"📊 Total words added to your wellness library: {total_words:,}")
        with st.expander("📋 View converted files"):
            for doc in converted_docs:
                st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
    if errors:
        st.error(f"❌ {len(errors)} file(s) failed to convert:")
        for error in errors:
            st.write(f"• {error}")


# --- Enhanced Q&A with source ---
def get_answer_with_source(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else ["unknown"] * len(docs)
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my Holistic Library.", "No source"
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    best_source = ids[0].split('_chunk_')[0] if ids else "unknown"
    return answer, best_source


# --- Search history ---
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    st.subheader("🕒 Your Gentle Search History")
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet. Your journey awaits! 🌱")
        return
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])


# --- Document manager ---
def show_document_manager():
    st.subheader("📚 Your Wellness Library")
    if not st.session_state.get('converted_docs'):
        st.info("Your library is waiting for nourishing wisdom. Add a document to begin! 🌱")
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("🌿 Preview this Wisdom", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("🧘‍♂️ Release from My Library", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                st.session_state.collection = reset_collection(chromadb.Client(), "documents")
                add_docs_to_database(st.session_state.collection, st.session_state.converted_docs)
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("🌙 Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()


# --- Document statistics ---
def show_document_stats():
    st.subheader("📊 Holistic Document Insights")
    if not st.session_state.get('converted_docs'):
        st.info("No documents to analyze yet. Add some wisdom to see your holistic stats! 🌼")
        return
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    st.write("**File Types in Your Wellness Library:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} file{'s' if count > 1 else ''}")


# --- Helper: Add docs to ChromaDB ---
def add_docs_to_database(collection, docs):
    for doc in docs:
        add_text_to_chromadb(doc['content'], doc['filename'], collection_name="documents")
    return len(docs)


# --- Main holistic app ---
def holistic_main():
    add_holistic_css()
    st.title("Hello, welcome to Holistica! 🧘🏻‍♀️🌀")
    st.markdown(
        """
        <div style="background:rgba(255,255,240,0.7);border-radius:14px;padding:1.2em 1.5em 1.2em 1.5em;margin-bottom:1.5em;">
        <span style="font-size:1.2em; color:#7a4c1e;">
        Discover a world where healing goes beyond medicine.<br>
        <b>This app is your cozy space for exploring the five dimensions of holistic health:</b><br><br>
        <ul style="line-height:1.5;margin-left:1.2em;">
        <li>Physical <span style="font-size:1.1em;">🏃‍♀️</span></li>
        <li>Mental <span style="font-size:1.1em;">🧠</span></li>
        <li>Emotional <span style="font-size:1.1em;">❤️‍🩹</span></li>
        <li>Spiritual <span style="font-size:1.1em;">🧿</span></li>
        <li>Social/Environmental <span style="font-size:1.1em;">🫂💚</span></li>
        </ul>
        <i>Let’s nurture your well-being together, one gentle step at a time.</i>
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Session state
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'collection' not in st.session_state:
        st.session_state.collection = chromadb.Client().create_collection(name="documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌱 Upload Wellness Wisdom",
        "💬 Gentle Q&A",
        "📋 Your Library",
        "📊 Insights & Balance"
    ])
    with tab1:
        st.header("Upload & Convert Your Wellness Documents")
        uploaded_files = st.file_uploader(
            "Bring your knowledge into our cozy space (PDF, DOC, DOCX, TXT)",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True
        )
        if st.button("✨ Add to My Holistic Library ✨"):
            if uploaded_files:
                converted_docs, errors = safe_convert_files(uploaded_files)
                if converted_docs:
                    num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                    st.session_state.converted_docs.extend(converted_docs)
                show_conversion_results(converted_docs, errors)
    with tab2:
        st.header("Ask a Gentle Question")
        if st.session_state.get('converted_docs'):
            question = st.text_input("What would you like to explore on your wellness journey today?")
            if st.button("🌸 Find My Holistic Answer 🌸"):
                if question:
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                    st.write("**Answer:**")
                    st.write(answer)
                    st.write(f"**Source:** {source}")
                    add_to_search_history(question, answer, source)
        else:
            st.info("Please add some documents to your library first! 🌿")
        show_search_history()
    with tab3:
        st.header("Your Wellness Library")
        show_document_manager()
    with tab4:
        st.header("Holistic Insights & Balance")
        show_document_stats()
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by AI*")


# MAIN APP
def main():
    st.title("📚 Smart Document Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )

    client = chromadb.Client()

    if st.button("Chunk and Store Documents"):
        if uploaded_files:
            collection = reset_collection(client, "documents")
            for file in uploaded_files:
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(file.getvalue())
                    temp_file_path = temp_file.name

                text = convert_to_markdown(temp_file_path)
                collection = add_text_to_chromadb(text, file.name, collection_name="documents")
                st.write(f"Stored {file.name} in ChromaDB")
        else:
            st.error("Upload files first!")

    question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        try:
            collection = client.get_collection(name="documents")
            answer = get_answer(collection, question)
            st.write("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    holistic_main()
