import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_postgres import PGVector
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


def get_chat_model() -> str:
    return os.getenv("GOOGLE_CHAT_MODEL", "").strip() or "gemini-3.5-flash-lite"

FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."
RESULTS_COUNT = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _create_vector_store(api_key: str) -> PGVector:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=get_embedding_model(),
        google_api_key=api_key,
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )


def _search_documents(
    vector_store: PGVector, question: str, k: int = RESULTS_COUNT
) -> list[tuple[Document, float]]:
    question = question.strip()
    if not question:
        raise ValueError("A pergunta não pode estar vazia.")
    if k < 1:
        raise ValueError("O número de resultados deve ser maior que zero.")

    return vector_store.similarity_search_with_score(question, k=k)


def search_documents(
    question: str, k: int = RESULTS_COUNT
) -> list[tuple[Document, float]]:
    vector_store = _create_vector_store(required_env("GOOGLE_API_KEY"))
    return _search_documents(vector_store, question, k)


def _format_context(results: list[tuple[Document, float]]) -> str:
    passages = []
    for document, _score in results:
        page = document.metadata.get("page")
        page_label = f" (página {page + 1})" if isinstance(page, int) else ""
        passages.append(f"[Fonte{page_label}]\n{document.page_content}")
    return "\n\n".join(passages)


def search_prompt(question: str | None = None):
    api_key = required_env("GOOGLE_API_KEY")
    vector_store = _create_vector_store(api_key)
    llm = ChatGoogleGenerativeAI(
        model=get_chat_model(),
        google_api_key=api_key,
        temperature=0,
    )

    def answer(inputs: dict[str, str]) -> str:
        user_question = inputs.get("question", "").strip()
        if not user_question:
            raise ValueError("A pergunta não pode estar vazia.")

        results = _search_documents(vector_store, user_question)
        if not results:
            return FALLBACK_ANSWER

        prompt = PROMPT_TEMPLATE.format(
            contexto=_format_context(results),
            pergunta=user_question,
        )
        return (llm | StrOutputParser()).invoke(prompt)

    chain = RunnableLambda(answer)
    if question is None:
        return chain
    return chain.invoke({"question": question})