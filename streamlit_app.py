import os
import faiss
import numpy as np
import streamlit as st
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from rag import load_pdf_with_pages, split_pages_into_chunks

# PAGE CONFIGURATION

st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="centered"
)

# CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_source" not in st.session_state:
    st.session_state.active_source = None

# LOAD ENVIRONMENT VARIABLES

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()

# TITLE
st.title("📚 PDF Q&A Assistant")
st.write(
    "Ask questions about your PDF and get answers "
    "using Retrieval-Augmented Generation (RAG)."
)
# PDF UPLOAD

st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)
# CHECK WHETHER PDF WAS UPLOADED
if not uploaded_files:
    st.info(
        "Please upload at least one PDF from the sidebar to begin."
    )
    st.stop()

# PROCESS UPLOADED PDFs
os.makedirs("data", exist_ok=True)
all_chunks = []
for uploaded_file in uploaded_files:
    pdf_path = f"data/{uploaded_file.name}"
    # Save uploaded PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Extract pages
    pages = load_pdf_with_pages(pdf_path)
    # Split pages into chunks
    file_chunks = split_pages_into_chunks(pages)
    # Store chunk + metadata
    for chunk in file_chunks:
        all_chunks.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "source": uploaded_file.name
        })
#store the chunks
chunks = all_chunks
print("\n--- DEBUG: CONCLUSION CHUNKS ---")
for i, chunk in enumerate(chunks):
    if (
        chunk["source"] == "sample.pdf"
        and chunk["page"] == 19
    ):
        print(f"\nChunk index: {i}")
        print("Text:", chunk["text"][:800])

# SIDEBAR DOCUMENT INFORMATION

st.sidebar.success(
    f"{len(uploaded_files)} PDF(s) loaded successfully!"
)
st.sidebar.write(
    f"**Documents:** {len(uploaded_files)}"
)
st.sidebar.write(
    f"**Total Chunks:** {len(chunks)}"
)
st.sidebar.write("****Files:****")
for uploaded_file in uploaded_files:
    st.sidebar.write(
        f"📄 {uploaded_file.name}"
    )
# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
embedding_model = load_embedding_model()

# LOAD CROSS-ENCODER RERANKER

@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
reranker = load_reranker()

# CREATE FAISS VECTOR STORE
def create_vector_store(chunks):
    # Extract chunk text
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]
    # Create embeddings
    embeddings = embedding_model.encode(
        chunk_texts,
        normalize_embeddings=True
    )
    # Convert to float32 for FAISS
    embeddings = np.array(
        embeddings
    ).astype("float32")
    # Embedding dimension
    dimension = embeddings.shape[1]
    # Inner Product index
    # With normalized embeddings this behaves
    # like cosine similarity
    index = faiss.IndexFlatIP(
        dimension
    )
    # Add embeddings
    index.add(embeddings)
    return index
index = create_vector_store(chunks)
# CREATE SOURCE-SPECIFIC FAISS INDEXES

source_faiss_indexes = {}
source_chunk_indices = {}

for source in set(chunk["source"] for chunk in chunks):

    # Get original chunk indexes for this PDF.
    global_indices = [
        i for i, chunk in enumerate(chunks)
        if chunk["source"] == source
    ]

    source_texts = [
        chunks[i]["text"] for i in global_indices
    ]

    # Generate embeddings for this PDF.
    source_embeddings = embedding_model.encode(
        source_texts,
        normalize_embeddings=True
    )

    source_embeddings = np.asarray(
        source_embeddings,
        dtype=np.float32
    )

    # Create FAISS index.
    source_index = faiss.IndexFlatIP(
        source_embeddings.shape[1]
    )

    source_index.add(source_embeddings)

    source_faiss_indexes[source] = source_index
    source_chunk_indices[source] = global_indices

# LOAD GEMINI
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=api_key,
        temperature=0
    )
llm = load_llm()

# DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.write(
            message["content"]
        )
# CHAT INPUT
question = st.chat_input(
    "Ask something about your document..."
)

# PROCESS QUESTION
if question:
    question_lower = question.lower()

    # Detect an explicitly mentioned project/document.
    if (
        "aroma avenue" in question_lower
        or "perfume website" in question_lower
        or "perfume report" in question_lower
    ):
        st.session_state.active_source = "internship_report.pdf"

    elif (
        "hotel management system" in question_lower
        or "hotel management" in question_lower
    ):
        st.session_state.active_source = "sample.pdf"

    print(
    "DEBUG ACTIVE SOURCE:",
    st.session_state.active_source
    )
    #DISPLAY USER QUESTION
    with st.chat_message("user"):
        st.write(question)
    # SAVE USER MESSAGE
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    # RETRIEVAL
    with st.spinner(
        "Searching the document..."
    ):
        # CREATE QUESTION EMBEDDING
        question_embedding = embedding_model.encode(
            [question],
            normalize_embeddings=True
        )
        question_embedding = np.array(
            question_embedding
        ).astype("float32")
        # FAISS SEARCH

        active_source = st.session_state.active_source

        print(
            "DEBUG SEARCH SOURCE:",
            active_source
        )

        if active_source in source_faiss_indexes:

            # Search only the active PDF.
            search_index = source_faiss_indexes[active_source]

            global_indices = source_chunk_indices[active_source]

            k = min(20, len(global_indices))

            scores, local_indices = search_index.search(
                question_embedding,
                k
            )

            # Convert local FAISS indexes to original chunk indexes.
            indices = np.array([[
                global_indices[i] if i >= 0 else -1
                for i in local_indices[0]
            ]])

        else:

            # No active document: search all uploaded PDFs.
            k = min(20, len(chunks))

            scores, indices = index.search(
                question_embedding,
                k
            )
        # STORE RETRIEVED CHUNKS
        retrieved_chunks = []
        for i, index_number in enumerate(indices[0]):
            if index_number < 0 or index_number >= len(chunks):
                continue

            retrieved_chunks.append({
                "chunk": chunks[index_number]["text"],
                "page": chunks[index_number]["page"],
                "source": chunks[index_number]["source"],
                "score": float(scores[0][i]),
                "index": int(index_number),
            })

        # Supplement retrieval for conclusion / future-scope questions

        section_keywords = (
            "conclusion" in question_lower
            or "future scope" in question_lower
            or "future improvement" in question_lower
            or "limitations" in question_lower
        )

        # Initialize these for every question.
        heading_indices = []
        section_indices = []

        if section_keywords:
            heading_indices = [
                i for i, chunk in enumerate(chunks)
                if (
                    "conclusion and future scope" in chunk["text"].lower()
                    and (
                        active_source is None
                        or chunk["source"] == active_source
                    )
                )
            ]
            if heading_indices:
                heading_index = heading_indices[0]
                heading_chunk = chunks[heading_index]

                # Collect the section from the heading until references.
                for i in range(heading_index, len(chunks)):
                    chunk = chunks[i]

                    if chunk["source"] != heading_chunk["source"]:
                        break

                    text_lower = chunk["text"].lower()

                    if i > heading_index and (
                        "references" in text_lower[:250]
                        or "bibliography" in text_lower[:250]
                    ):
                        break

                    section_indices.append(i)

                existing_indices = {
                    item["index"] for item in retrieved_chunks
                }

                missing_indices = [
                    i for i in section_indices
                    if i not in existing_indices
                ]

                if missing_indices:
                    missing_embeddings = embedding_model.encode(
                        [chunks[i]["text"] for i in missing_indices],
                        normalize_embeddings=True
                    )

                    missing_embeddings = np.asarray(
                        missing_embeddings,
                        dtype=np.float32
                    )

                    similarities = (
                        missing_embeddings @ question_embedding[0]
                    )

                    for chunk_index, similarity in zip(
                        missing_indices, similarities
                    ):
                        retrieved_chunks.append({
                            "chunk": chunks[chunk_index]["text"],
                            "page": chunks[chunk_index]["page"],
                            "source": chunks[chunk_index]["source"],
                            "score": float(similarity),
                            "index": chunk_index,
                        })


                
        # HYBRID RETRIEVAL: KEYWORD-BASED SUPPLEMENT

        technology_question = any(
            term in question_lower
            for term in [
                "technology",
                "technologies",
                "programming language",
                "framework",
                "database",
                "library",
                "libraries",
                "tech stack",
                "tools used",
                "software used"
            ]
        )

        technology_candidate_indices = set()

        if technology_question:
            technology_terms = [
                "html", "css", "javascript",
                "reactjs", "react.js",
                "node.js", "nodejs",
                "express.js", "mongodb",
                "mysql", "java", "python",
                "php", "ruby", "angular",
                "vue.js", "django", "flask",
                "spring boot", "tensorflow",
                "pytorch", "opencv", "faiss",
                "langchain", "streamlit",
                "bootstrap", "jquery",
                "sql", "firebase",
                "software requirements",
                "hardware requirements",
                "development tools",
                "technologies used"
            ]

            existing_indices = {
                item["index"]
                for item in retrieved_chunks
            }

            keyword_candidates = []
            for i, chunk in enumerate(chunks):
                if active_source is not None and chunk["source"] != active_source:
                    continue

                chunk_text = chunk["text"].lower()

                matched_terms = [
                    term for term in technology_terms
                    if re.search(
                        rf"(?<!\w){re.escape(term)}(?!\w)",
                        chunk_text
                    )
                ]

                if matched_terms and i not in existing_indices:

                    keyword_candidates.append({
                        "chunk": chunk["text"],
                        "page": chunk["page"],
                        "source": chunk["source"],
                        "score": 0.0,
                        "index": i,
                        "keyword_matches": len(matched_terms)
                    })

            # Prioritize chunks containing more technology terms
            keyword_candidates.sort(
                key=lambda x: x["keyword_matches"],
                reverse=True
            )
            # Remember the indexes of the 10 best keyword-matched chunks.
            technology_candidate_indices = {
                item["index"]
                for item in keyword_candidates[:10]
            }

            # Add up to 10 additional chunks
            for item in keyword_candidates[:10]:
                retrieved_chunks.append(item)

            print("\n--- DEBUG: KEYWORD RETRIEVAL ---")

            for item in keyword_candidates[:10]:
                print(
                    f"Source: {item['source']} | "
                    f"Page: {item['page']} | "
                    f"Matches: {item['keyword_matches']}"
                )
                print(item["chunk"][:300])
        
        # FINAL SOURCE FILTER
        if active_source is not None:
            retrieved_chunks = [
                item for item in retrieved_chunks
                if item["source"] == active_source
            ]    

        if not retrieved_chunks:

            st.warning(

                "I could not find relevant information "

                "in the document."

            )

            st.stop()

        best_faiss_score = max(item["score"] for item in retrieved_chunks)
        # CROSS-ENCODER RERANKING

        rerank_pairs = [
            [question, item["chunk"]]
            for item in retrieved_chunks
        ]
        rerank_scores = reranker.predict(
            rerank_pairs
        )
        for item, rerank_score in zip(
            retrieved_chunks,
            rerank_scores
        ):
            item["rerank_score"] = float(
                rerank_score
            )
        # NORMALIZE SCORES
        faiss_scores = np.array([
            item["score"]
            for item in retrieved_chunks
        ])
        reranker_scores = np.array([
            item["rerank_score"]
            for item in retrieved_chunks
        ])
        # Min-Max normalization
        # Converts both score types approximately to 0-1
        if faiss_scores.max() != faiss_scores.min():
            normalized_faiss = (
                (faiss_scores - faiss_scores.min())
                /
                (faiss_scores.max() - faiss_scores.min())
            )
        else:
            normalized_faiss = np.ones(
                len(faiss_scores)
            )
        if (
            reranker_scores.max()
            != reranker_scores.min()
        ):
            normalized_reranker = (
                (reranker_scores - reranker_scores.min())
                /
                (
                    reranker_scores.max()
                    - reranker_scores.min()
                )
            )
        else:
            normalized_reranker = np.ones(
                len(reranker_scores)
            )

        # SCORE FUSION
        faiss_weight = 0.4
        reranker_weight = 0.6
        for i, item in enumerate(
            retrieved_chunks
        ):
            item["normalized_faiss"] = float(
                normalized_faiss[i]
            )
            item["normalized_reranker"] = float(
                normalized_reranker[i]
            )
            item["combined_score"] = (
                faiss_weight
                * item["normalized_faiss"]
                +
                reranker_weight
                * item["normalized_reranker"]
            )
        # RETRIEVAL DEBUG
        print("\n--- DEBUG: ALL RETRIEVAL CANDIDATES ---")
        for i, item in enumerate(retrieved_chunks):
            print(f"\nCandidate {i + 1}")
            print("Source:", item["source"])
            print("Page:", item["page"])
            print("FAISS score:", round(item["score"], 4))
            print("Text:", item["chunk"][:300])

        # SORT BY COMBINED SCORE
        retrieved_chunks = sorted(
            retrieved_chunks,
            key=lambda x: x["combined_score"],
            reverse=True
        )
        print("\n--- DEBUG: AFTER RERANKING ---")
        for i, item in enumerate(retrieved_chunks):
            print(f"\nRank {i + 1}")
            print("Page:", item["page"])
            print("FAISS:", round(item["score"], 4))
            print("Reranker:", round(item["rerank_score"], 4))
            print("Combined:", round(item["combined_score"], 4))
            print("Text:", item["chunk"][:200])
        # SELECT FINAL CONTEXT CHUNKS
        if section_keywords and heading_indices:

            section_index_set = set(section_indices)

            section_chunks = [
                item for item in retrieved_chunks
                if item["index"] in section_index_set
            ]

            unique_chunks = {}

            for item in section_chunks:
                unique_chunks[item["index"]] = item

            retrieved_chunks = sorted(
                unique_chunks.values(),
                key=lambda item: item["index"]
            )

        elif technology_question:

            # Keep the 5 highest-ranked chunks.
            selected_chunks = retrieved_chunks[:5]

            selected_indices = {
                item["index"] for item in selected_chunks
            }

            # Add technology-related chunks that may have ranked lower.
            for item in retrieved_chunks:

                if item["index"] in technology_candidate_indices:
                    if item["index"] not in selected_indices:

                        selected_chunks.append(item)
                        selected_indices.add(item["index"])

                if len(selected_chunks) >= 10:
                    break

            # Restore original document order.
            retrieved_chunks = sorted(
                selected_chunks,
                key=lambda item: item["index"]
            )

        else:

            # Normal questions: retain the top 5 chunks.
            retrieved_chunks = retrieved_chunks[:5]
        

        print("\n--- DEBUG: FINAL CONTEXT SENT TO GEMINI ---")
        for i, item in enumerate(retrieved_chunks, start=1):
            print(f"\nSource {i} | Page {item['page']}")
            print(item["chunk"])
        # CHECK WHETHER THE RETRIEVED CONTEXT IS RELEVANT

        MIN_RELEVANCE_SCORE = 0.30

        if best_faiss_score < MIN_RELEVANCE_SCORE:

            st.warning(
                f"Low FAISS similarity: {best_faiss_score:.3f}. "
                "Checking the retrieved context before deciding."
            )

            answer = "I could not find the answer in the document."
            with st.chat_message("assistant"):
                st.write(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            st.stop()
        # BUILD CONTEXT
        context = "\n\n".join(
            item["chunk"]
            for item in retrieved_chunks
        )
        if technology_question:
            st.caption(
                f"Retrieved up to {k} FAISS candidates → "
                "reranked with keyword matching → "
                f"using {len(retrieved_chunks)} context chunks."
            )
        else:
            st.caption(
                f"Retrieved up to {k} candidates → "
                "reranked → using top 5 sources."
            )
    # GENERATE ANSWER
    with st.spinner(
        "Generating answer..."
    ):
        # BUILD CONVERSATION HISTORY
        conversation_history = ""
        for message in st.session_state.messages[:-1]:
            # Skip previous fallback answers
            if (
                message["role"] == "assistant"
                and message["content"].strip()
                == "I could not find the answer in the document."
            ):
                continue
            conversation_history += (
                f"{message['role'].upper()}: "
                f"{message['content']}\n"
    )
        # PROMPT
        prompt = f"""
You are a helpful document question-answering assistant.
You answer questions using the provided document context
and the conversation history.
IMPORTANT RULES:
1. Use ONLY information supported by the document context.

2. You may use conversation history to understand references

   such as "they", "their", "it", "this project", etc.

3. Do not invent information.

4. If the answer cannot be found in the document context,

   say exactly:

   "I could not find the answer in the document."

5. If the context identifies the project's programming language or describes it as a project developed in that language, answer directly using that evidence. For example, if the document describes the project as "Hotel Management System in Java" and explains Java in the software requirements, answer that Java was used.

6. Do not require separate confirmation of implementation when the document provides sufficient evidence to identify the technology used in the project. Distinguish between a technology merely mentioned in passing and one described as part of the project's design, requirements, or implementation.

7. Do not reject a question merely because the evidence is incomplete. Explain what the retrieved context establishes and clearly state any uncertainty.

8. Use the exact fallback response only when the provided context contains no useful information relevant to the question. Never invent missing information or use outside knowledge to fill gaps.

9. Answer the current question directly and concisely.

Conversation History:

{conversation_history}

Document Context:

{context}

Current Question:

{question}

Answer:

"""
        # CALL GEMINI
        response = llm.invoke(
            prompt
        )
    # EXTRACT ANSWER
    answer = response.content
    if isinstance(
        answer,
        list
    ):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
            and item.get("type") == "text"
        )
    # DISPLAY ANSWER
    with st.chat_message(
        "assistant"
    ):
        st.write(
            answer
        )
        # DISPLAY SOURCES
        with st.expander(
            "📚 View Sources"
        ):
            for i, item in enumerate(
                retrieved_chunks
            ):
                st.markdown(
                    f"**Source {i + 1}**"
                )
                st.caption(f"{item['source']} | Page {item['page']}")
                st.write(
                    item["chunk"]
                )
                st.divider()
    # SAVE ASSISTANT RESPONSE
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })