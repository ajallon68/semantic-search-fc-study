import os
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

Provider = Literal["google", "openai"]


def get_provider() -> Provider:
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if google_key and openai_key:
        raise ValueError(
            "Configure apenas uma chave de provedor: GOOGLE_API_KEY ou "
            "OPENAI_API_KEY, não ambas."
        )
    if openai_key:
        return "openai"
    if google_key:
        return "google"
    raise ValueError(
        "Configure GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env."
    )


def create_embeddings(provider: Provider):
    if provider == "openai":
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
            or "text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        )
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "").strip()
        or "models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
    )


def get_collection_name(provider: Provider) -> str:
    base_name = (
        os.getenv("PG_VECTOR_COLLECTION_NAME", "").strip() or "pdf_documents"
    )
    return f"{base_name}_openai" if provider == "openai" else base_name


def get_database_url() -> str:
    return (
        os.getenv("DATABASE_URL", "").strip()
        or "postgresql+psycopg://postgres:postgres@localhost:5433/rag"
    )


def get_pdf_path() -> Path:
    value = os.getenv("PDF_PATH", "document.pdf").strip() or "document.pdf"
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ingest_pdf() -> int:
    pdf_path = get_pdf_path()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"O arquivo informado não é um PDF: {pdf_path}")

    provider = get_provider()
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

    vector_store = PGVector(
        embeddings=create_embeddings(provider),
        collection_name=get_collection_name(provider),
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
