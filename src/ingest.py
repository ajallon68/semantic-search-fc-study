import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"A variável {name} não está configurada. "
            "Copie .env.example para .env e informe os valores necessários."
        )
    return value


def get_pdf_path() -> Path:
    value = os.getenv("PDF_PATH", "document.pdf").strip() or "document.pdf"
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_database_url() -> str:
    return (
        os.getenv("DATABASE_URL", "").strip()
        or "postgresql+psycopg://postgres:postgres@localhost:5433/rag"
    )


def get_collection_name() -> str:
    return os.getenv("PG_VECTOR_COLLECTION_NAME", "").strip() or "pdf_documents"


def get_embedding_model() -> str:
    return (
        os.getenv("GOOGLE_EMBEDDING_MODEL", "").strip()
        or "models/gemini-embedding-001"
    )


def ingest_pdf() -> int:
    pdf_path = get_pdf_path()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"O arquivo informado não é um PDF: {pdf_path}")

    api_key = required_env("GOOGLE_API_KEY")
    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError(f"O PDF não contém texto para indexar: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"Não foi possível gerar chunks a partir do PDF: {pdf_path}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=get_embedding_model(),
        google_api_key=api_key,
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )
    source = str(pdf_path.resolve())
    ids = [
        str(
            uuid5(
                NAMESPACE_URL,
                f"{source}:{chunk.metadata.get('page', 0)}:{chunk_index}",
            )
        )
        for chunk_index, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(documents=chunks, ids=ids)

    print(f"Ingestão concluída: {len(chunks)} chunks salvos da origem {pdf_path.name}.")
    return len(chunks)


if __name__ == "__main__":
    ingest_pdf()