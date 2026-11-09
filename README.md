# Ingestão e busca semântica de PDFs

Aplicação de perguntas e respostas sobre o conteúdo de um PDF. O texto é
dividido em trechos, convertido em embeddings e armazenado no PostgreSQL com a
extensão pgvector. A geração de embeddings e respostas pode usar Google Gemini
ou OpenAI, conforme a chave configurada no `.env`.

## Scripts

- `src/ingest.py`: extrai o texto do PDF, divide-o em trechos e salva os
  embeddings no PostgreSQL, usando Gemini ou OpenAI.
- `src/search.py`: consulta os trechos mais relevantes e prepara uma resposta
  com o provedor selecionado. Também oferece `search_documents(question, k=10)`
  para fazer buscas sem gerar uma resposta.
- `src/chat.py`: inicia uma interface de linha de comando para fazer perguntas
  sobre o documento.

## Estrutura do projeto

```text
├── docker-compose.yml
├── requirements.txt    # Dependencies
├── .env.example        # Environment variables template
├── src/
│   ├── ingest.py       # Script - Ingest PDF
│   ├── search.py       # Script - Search
│   └── chat.py         # CLI for user interaction
├── document.pdf        # PDF to be used
└── README.md
```

## Execução

Execute os comandos a seguir a partir da raiz do projeto.

### 1. Instalar as dependências

Requisitos: Python 3.10 ou superior, Docker e Docker Compose.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows, ative o ambiente com `venv\Scripts\activate`.

### 2. Configurar as variáveis de ambiente

Crie o arquivo `.env` a partir do modelo:

```bash
cp .env.example .env
```

Edite `.env` e preencha **apenas uma** das chaves: `GOOGLE_API_KEY` para Google
Gemini ou `OPENAI_API_KEY` para OpenAI. O programa escolhe o provedor pela
chave preenchida e interrompe a execução com uma mensagem clara se nenhuma
chave, ou as duas chaves, estiverem configuradas. Não compartilhe o `.env` nem
envie suas chaves ao Git.

| Variável | Descrição | Valor padrão |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Chave para acessar os modelos do Google | Sem valor; obrigatório |
| `GOOGLE_EMBEDDING_MODEL` | Modelo usado para gerar embeddings | `models/gemini-embedding-001` |
| `GOOGLE_CHAT_MODEL` | Modelo de linguagem usado para responder | `gemini-3.5-flash-lite` |
| `OPENAI_API_KEY` | Chave para acessar os modelos da OpenAI | Sem valor |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings da OpenAI | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Modelo de chat da OpenAI | `gpt-4o-mini` |
| `DATABASE_URL` | Endereço de conexão ao PostgreSQL local | `******localhost:5433/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção vetorial | `pdf_documents` |
| `PDF_PATH` | Caminho do PDF para ingestão | `document.pdf` |

`PDF_PATH` pode ser absoluto ou relativo à raiz do projeto. PDFs compostos
apenas por páginas digitalizadas podem precisar de OCR antes da ingestão.
As coleções pgvector são separadas por provedor (`pdf_documents` para Gemini e
`pdf_documents_openai` para OpenAI). O valor de `PG_VECTOR_COLLECTION_NAME` é
usado como nome base; ao usar OpenAI, o sistema acrescenta `_openai`. Ao trocar
de provedor ou modelo de embeddings, execute a ingestão novamente; os vetores
não são compatíveis entre modelos.

### 3. Iniciar o banco de dados

```bash
docker compose up -d
docker compose ps
```

O PostgreSQL fica disponível na porta `5433` da máquina. O Compose inicializa
a extensão pgvector automaticamente.

### 4. Ingerir o PDF

```bash
python src/ingest.py
```

A ingestão usa `document.pdf` por padrão. Para processar outro arquivo, ajuste
`PDF_PATH` no `.env`. A primeira execução precisa acessar a API do provedor
selecionado para gerar os embeddings.

### 5. Fazer perguntas pelo chat

```bash
python src/chat.py
```

Digite uma pergunta quando aparecer `PERGUNTA:`. O chat permanece aberto até
que você digite `sair`, `exit` ou `quit`. Para cada pergunta, o sistema recupera
até 10 trechos do banco e gera uma resposta baseada neles. Se o contexto não
contiver a informação necessária, a resposta será
`Não tenho informações necessárias para responder sua pergunta.`

## Encerrar o banco

Para parar os serviços sem apagar os dados:

```bash
docker compose down
```

Os dados ficam no volume `postgres_data`. Não use `docker compose down -v` a
menos que queira remover também o banco e os dados ingeridos.

## Observação

Os vetores de uma coleção devem ser compatíveis com o modelo de embeddings
usado para gerá-los. Se trocar o provedor ou o modelo de embeddings por outro
com dimensão diferente, remova a coleção correspondente ou use outra e faça a
ingestão novamente.
