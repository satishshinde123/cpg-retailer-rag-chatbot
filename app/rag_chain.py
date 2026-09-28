import logging
from pathlib import Path

from langchain_groq import ChatGroq

from app.config import (
    GROQ_API_KEY,
    LLM_MODEL
)

from app.retriever import (
    retrieve_documents
)


logger = logging.getLogger(__name__)


# --------------------------------------------------
# LLM
# --------------------------------------------------

def get_llm():

    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY
    )


# --------------------------------------------------
# Format context
# --------------------------------------------------

def format_context(documents):

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            None
        )

        if isinstance(page, int):

            page_number = page + 1

        else:

            page_number = "Unknown"

        filename = Path(
            source
        ).name

        context_parts.append(
            f"""
SOURCE DOCUMENT:
{filename}

PAGE:
{page_number}

CONTENT:
{document.page_content}
"""
        )

    return "\n".join(
        context_parts
    )


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

def generate_answer(
    question,
    documents
):

    context = format_context(
        documents
    )

    prompt = f"""
You are a CPG and Retail Business
Intelligence Assistant.

You answer questions ONLY using the
provided CPG/Retail document context.

IMPORTANT RULES:

1. Use only information contained in
   the retrieved documents.

2. Do not use outside knowledge.

3. Do not invent prices, quantities,
   policies, dates, products or numbers.

4. If the answer is not present in
   the context, respond:

   "I could not find this information
   in the provided CPG documents."

5. Give a clear and concise answer.

6. When answering numerical questions,
   preserve the exact values from
   the documents.

7. Do not combine information from
   unrelated documents unless necessary.

8. Do not make assumptions.

Retrieved Context
=================

{context}

=================

User Question
=============

{question}

Answer:
"""

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    return response.content


# --------------------------------------------------
# RAG
# --------------------------------------------------

def ask_question(question):

    documents = retrieve_documents(
        question
    )

    if not documents:

        return (
            "I could not find relevant "
            "information in the provided "
            "CPG documents."
        ), []

    answer = generate_answer(
        question,
        documents
    )

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            None
        )

        if isinstance(page, int):

            page = page + 1

        filename = Path(
            source
        ).name

        source_info = (
            f"{filename} - Page {page}"
        )

        if source_info not in sources:

            sources.append(
                source_info
            )

    return answer, sources