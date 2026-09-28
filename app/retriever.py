import logging

from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from app.config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    TOP_K
)


logger = logging.getLogger(__name__)


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_vector_store():

    if not VECTOR_STORE_DIR.exists():

        raise FileNotFoundError(
            "FAISS index not found. "
            "Run ingestion first."
        )

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_documents(
    query,
    top_k=TOP_K
):

    vector_store = load_vector_store()

    logger.info(
        "Retrieving top %d documents for: %s",
        top_k,
        query
    )

    documents = vector_store.similarity_search(
        query,
        k=top_k
    )

    return documents