"""End-to-end: load a T4 PDF via FinLitLoader, split, embed, query.

Requires: pip install finlit[langchain] langchain-openai langchain-chroma
Env: ANTHROPIC_API_KEY (for FinLit's Claude extractor), OPENAI_API_KEY
     (for embeddings).
"""
from __future__ import annotations

import sys
from pathlib import Path

from finlit.integrations.langchain import FinLitLoader


def main(paths: list[str]) -> None:
    # Load in batch with on_error='include' so compliance teams see every
    # file that was submitted, even failed ones.
    loader = FinLitLoader(paths, schema="cra.t4", on_error="include")
    docs = loader.load()

    # Filter out failure Documents before feeding an embedder (empty
    # page_content will cause some embedders to error out).
    good_docs = [d for d in docs if not d.metadata.get("finlit_error")]
    print(f"Loaded {len(good_docs)} successful, {len(docs) - len(good_docs)} failed")

    # Structured field access — no embedding needed for this kind of query
    for d in good_docs:
        fields = d.metadata["finlit_fields"]
        needs_review = d.metadata["finlit_needs_review"]
        print(
            f"{Path(d.metadata['source']).name}: "
            f"{fields.get('employer_name')!r} "
            f"income={fields.get('box_14_employment_income')} "
            f"review={needs_review}"
        )

    # The same Documents are ready for a vector store. Uncomment to run
    # a real RAG pipeline (requires the extra deps above):
    #
    # from langchain_openai import OpenAIEmbeddings
    # from langchain_chroma import Chroma
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = splitter.split_documents(good_docs)
    # store = Chroma.from_documents(chunks, OpenAIEmbeddings())
    # hits = store.similarity_search("how much CPP did Acme withhold?")
    # for h in hits:
    #     print(h.page_content, h.metadata["finlit_fields"].get("box_16_cpp_contributions"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/langchain_rag.py <t4.pdf> [<t4.pdf> ...]")
        sys.exit(1)
    main(sys.argv[1:])
