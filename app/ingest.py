import logging
import shutil

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader
)

from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from app.config import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    CHUNKS_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Load PDFs
# --------------------------------------------------

def load_documents():

    logger.info(
        "Loading PDFs from: %s",
        DATA_DIR
    )

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    pdf_files = list(
        DATA_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF files found in data directory."
        )

    logger.info(
        "Found %d PDF files.",
        len(pdf_files)
    )

    loader = PyPDFDirectoryLoader(
        str(DATA_DIR)
    )

    documents = loader.load()

    logger.info(
        "Loaded %d PDF pages.",
        len(documents)
    )

    return documents


# --------------------------------------------------
# Split documents
# --------------------------------------------------

def split_documents(documents):

    logger.info(
        "Splitting documents..."
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    logger.info(
        "Created %d chunks.",
        len(chunks)
    )

    return chunks


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

def create_embeddings():

    logger.info(
        "Loading HuggingFace model: %s",
        EMBEDDING_MODEL
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# --------------------------------------------------
# Save chunks
# --------------------------------------------------

def save_chunks(chunks):

    chunk_file = (
        CHUNKS_DIR / "chunks.txt"
    )

    with open(
        chunk_file,
        "w",
        encoding="utf-8"
    ) as file:

        for i, chunk in enumerate(chunks):

            source = chunk.metadata.get(
                "source",
                "Unknown"
            )

            page = chunk.metadata.get(
                "page",
                "Unknown"
            )

            file.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            file.write(
                f"Chunk: {i}\n"
            )

            file.write(
                f"Source: {source}\n"
            )

            file.write(
                f"Page: {page}\n"
            )

            file.write(
                "=" * 80
                + "\n"
            )

            file.write(
                chunk.page_content
            )

            file.write("\n")


# --------------------------------------------------
# Create FAISS
# --------------------------------------------------

def create_vector_store(
    chunks,
    embeddings
):

    logger.info(
        "Creating FAISS vector store..."
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Remove previous index
    if VECTOR_STORE_DIR.exists():

        shutil.rmtree(
            VECTOR_STORE_DIR
        )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )

    logger.info(
        "FAISS index saved at: %s",
        VECTOR_STORE_DIR
    )

    return vector_store


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def ingest():

    try:

        logger.info(
            "===================================="
        )

        logger.info(
            "CPG RAG INGESTION STARTED"
        )

        logger.info(
            "===================================="
        )

        documents = load_documents()

        chunks = split_documents(
            documents
        )

        save_chunks(
            chunks
        )

        embeddings = create_embeddings()

        create_vector_store(
            chunks,
            embeddings
        )

        logger.info(
            "===================================="
        )

        logger.info(
            "INGESTION COMPLETED SUCCESSFULLY"
        )

        logger.info(
            "Documents/pages: %d",
            len(documents)
        )

        logger.info(
            "Chunks: %d",
            len(chunks)
        )

        logger.info(
            "===================================="
        )

    except FileNotFoundError as error:

        logger.error(
            "File error: %s",
            error
        )

    except Exception as error:

        logger.exception(
            "Unexpected ingestion error: %s",
            error
        )


if __name__ == "__main__":
    ingest()