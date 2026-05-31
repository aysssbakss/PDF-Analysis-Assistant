import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_DIR = "chroma_db"
UPLOAD_DIR = "pdfler"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="PDF RAG Asistanı",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.2
    )

embedding = get_embedding()
llm = get_llm()

st.title("🧠 PDF Analiz Asistanı")

with st.sidebar:

    st.header("PDF Yükleme")

    files = st.file_uploader(
        "PDF seç",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Hafızayı Güncelle"):

        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding
        )

        for file in files:

            file_path = os.path.join(
                UPLOAD_DIR,
                file.name
            )

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            docs = PyPDFLoader(file_path).load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            db.add_documents(chunks)

        st.success("PDF'ler eklendi")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Belge hakkında soru sor..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    context = ""

    for i, doc in enumerate(docs):
        kaynak = os.path.basename(
            doc.metadata.get("source", "Bilinmeyen")
        )

        sayfa = doc.metadata.get("page", 0) + 1

        context += f"""
    [KAYNAK {i + 1}]
    Dosya: {kaynak}
    Sayfa: {sayfa}

    {doc.page_content}

    """

    prompt = f"""
    Sen bir belge analiz uzmanısın.

    Aşağıdaki kaynakları kullanarak cevap ver.

    Kaynaklar:

    {context}

    Kurallar:

    - Sadece verilen kaynaklardan cevap ver.
    - Bilgi yoksa "Kaynaklarda bulunamadı" de.
    - Cevabın sonunda kullandığın kaynakları listele.

    Soru:
    {question}
    """

    response = llm.invoke(prompt)

    answer = response.content

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.markdown("## 📚 Kaynaklar")

    for doc in docs:
        dosya = os.path.basename(
            doc.metadata["source"]
        )

        sayfa = doc.metadata["page"] + 1

        with st.expander(
                f"{dosya} - Sayfa {sayfa}"
        ):
            st.write(doc.page_content[:1000])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
