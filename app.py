import os
import time
import uuid
import streamlit as st
from typing import List, Dict, Any

# Import reusable logic from the CLI chatbot
from chatbot import (
    load_config,
    build_client,
    build_openai_client,
    build_pinecone,
    embed_query,
    reformulate_query_with_llm,
    retrieve_context,
    extract_text_from_metadata,
    load_conv_history,
    append_conv_history,
)


def seed_env_from_secrets():
    """Populate os.environ from Streamlit secrets if present."""
    if hasattr(st, "secrets"):
        s = st.secrets
        # Map secrets (skip missing)
        for key in [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_MAX_TOKENS",
            "ANTHROPIC_TEMPERATURE",
            "SYSTEM_PROMPT_PATH",
            "OPENAI_API_KEY",
            "OPENAI_EMBEDDING_MODEL",
            "PINECONE_API_KEY",
            "PINECONE_ENVIRONMENT",
            "PINECONE_INDEX_NAME",
            "PINECONE_NAMESPACE",
            "PINECONE_TOP_K",
            "HISTORY_MAX_TURNS",
            "CONV_HISTORY_PATH",
            "LOG_LEVEL",
        ]:
            try:
                if key in s and s[key] is not None:
                    os.environ[key] = str(s[key])
            except Exception:
                pass


def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = None
    if "client" not in st.session_state:
        st.session_state.client = None
    if "oai" not in st.session_state:
        st.session_state.oai = None
    if "pc_index" not in st.session_state:
        st.session_state.pc_index = None
    if "conv_history" not in st.session_state:
        st.session_state.conv_history = []  # list of {role, content}
    # Removed last_ref - no longer needed with LLM reformulation
    if "history_path" not in st.session_state:
        # Per-session JSONL history file (override via CONV_HISTORY_PATH if provided)
        base_path = os.environ.get("CONV_HISTORY_PATH")
        if base_path:
            st.session_state.history_path = base_path
        else:
            os.makedirs("history", exist_ok=True)
            session_id = uuid.uuid4().hex
            st.session_state.history_path = os.path.join(
                "history", f"session-{session_id}.jsonl"
            )
    if "show_debug" not in st.session_state:
        # Default debug from env SHOW_DEBUG ("1" enables)
        st.session_state.show_debug = os.environ.get("SHOW_DEBUG", "0") == "1"


def bootstrap_backends():
    if st.session_state.client is None:
        seed_env_from_secrets()
        cfg = load_config()
        st.session_state.cfg = cfg
        st.session_state.client = build_client(cfg["api_key"])
        st.session_state.oai = build_openai_client(cfg.get("openai_api_key"))
        st.session_state.pc_index = build_pinecone(cfg)


def do_rag_and_respond(user_input: str) -> str:
    cfg = st.session_state.cfg
    client = st.session_state.client
    oai = st.session_state.oai
    pc_index = st.session_state.pc_index

    max_turns = int(os.getenv("HISTORY_MAX_TURNS", "5"))

    # Build context via RAG using new LLM reformulation approach
    context_blocks: List[str] = []
    
    # Load conversation history for reformulation
    conv_history_for_llm = load_conv_history(st.session_state.history_path, max_turns)

    # Use LLM to reformulate query based on conversation history
    effective_query = reformulate_query_with_llm(client, user_input, conv_history_for_llm, cfg["model"])
    
    # Check if LLM answered directly from history
    if effective_query.startswith("ANSWER:"):
        # Extract the direct answer and skip RAG
        direct_answer = effective_query[7:].strip()
        if st.session_state.show_debug:
            st.info("⚡ Answered from conversation history (no RAG)")
        # Save to history
        st.session_state.conv_history.append({"role": "user", "content": user_input})
        append_conv_history(st.session_state.history_path, "user", user_input)
        st.session_state.conv_history.append({"role": "assistant", "content": direct_answer})
        append_conv_history(st.session_state.history_path, "assistant", direct_answer)
        return direct_answer
    
    # Check if LLM needs clarification
    if effective_query.startswith("CLARIFY:"):
        # Extract the clarification question
        clarification_msg = effective_query[8:].strip()
        if st.session_state.show_debug:
            st.info("❓ Requesting clarification")
        # Save to history
        st.session_state.conv_history.append({"role": "user", "content": user_input})
        append_conv_history(st.session_state.history_path, "user", user_input)
        st.session_state.conv_history.append({"role": "assistant", "content": clarification_msg})
        append_conv_history(st.session_state.history_path, "assistant", clarification_msg)
        return clarification_msg
    
    if st.session_state.show_debug:
        st.info(f"🔄 Query reformulated: '{user_input}' → '{effective_query}'")

    # Embeddings + retrieve
    retrieved_sku = None
    retrieved_product = None
    retrieved_brand = None
    
    if oai and pc_index:
        try:
            vec = embed_query(oai, cfg["embedding_model"], effective_query)
            matches, _ = retrieve_context(pc_index, vec, cfg["pc_namespace"], cfg["pc_top_k"], metadata_filter=None)
            
            # Extract SKU and product info from top match for context tracking
            if matches:
                top_match_md = getattr(matches[0], "metadata", None) or matches[0].get("metadata", {})
                retrieved_sku = (top_match_md.get("SKU") or top_match_md.get("sku") or 
                                top_match_md.get("id") or top_match_md.get("product_id") or "").strip()
                retrieved_product = (top_match_md.get("Product") or top_match_md.get("product") or 
                                   top_match_md.get("product_name") or top_match_md.get("name") or "").strip()
                retrieved_brand = (top_match_md.get("Brand") or top_match_md.get("brand") or "").strip()
                
                if st.session_state.show_debug and (retrieved_sku or retrieved_product):
                    st.info(f"📦 Retrieved: {retrieved_product or 'N/A'} | {retrieved_brand or 'N/A'} | SKU: {retrieved_sku or 'N/A'}")

            for m in matches:
                md = getattr(m, "metadata", None) or m.get("metadata", {})
                txt = extract_text_from_metadata(md)
                if txt:
                    context_blocks.append(txt)

        except Exception as e:
            if st.session_state.show_debug:
                st.warning(f"Retrieval failed: {e}")

    context_text = "\n\n---\n\n".join(context_blocks).strip()

    # Prompt building
    rag_instructions = (
        "You are a precise assistant. Answer ONLY using the provided context. "
        "If the answer isn't in the context, say 'I don't know' succinctly."
    )
    question_block = f"Question:\n{user_input}"
    
    # Add SKU/product metadata to history for reformulation context
    product_context_line = ""
    if retrieved_sku or retrieved_product:
        context_parts = []
        if retrieved_product:
            context_parts.append(f"Product: {retrieved_product}")
        if retrieved_brand:
            context_parts.append(f"Brand: {retrieved_brand}")
        if retrieved_sku:
            context_parts.append(f"SKU: {retrieved_sku}")
        if context_parts:
            product_context_line = f"\n[Retrieved: {', '.join(context_parts)}]"
    
    if context_text:
        user_payload = f"{question_block}{product_context_line}\n\nContext:\n{context_text}"
    else:
        user_payload = f"{question_block}{product_context_line}\n\nContext:\n[No relevant context retrieved]"

    # Assemble history window for the LLM from persistent session history
    history_window = load_conv_history(st.session_state.history_path, max_turns)
    # Store clean user input for UI display, but send full payload to LLM and history file
    st.session_state.conv_history.append({"role": "user", "content": user_input})
    append_conv_history(st.session_state.history_path, "user", user_payload)

    # Call Anthropic
    t0 = time.perf_counter()
    response = st.session_state.client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system=f"{cfg['system']}\n\n{rag_instructions}",
        messages=history_window + [{"role": "user", "content": user_payload}],
    )
    t1 = time.perf_counter()
    if st.session_state.show_debug:
        st.caption(f"LLM latency: {t1 - t0:.2f}s")

    # Extract assistant text
    parts: List[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    assistant_text = "\n".join(p for p in parts if p)

    if not assistant_text:
        assistant_text = "[Empty response]"

    st.session_state.conv_history.append({"role": "assistant", "content": assistant_text})
    append_conv_history(st.session_state.history_path, "assistant", assistant_text)
    return assistant_text


# ---------------- UI ----------------
st.set_page_config(page_title="Lipstick QnA Chatbot", page_icon="💄", layout="centered")
init_state()
bootstrap_backends()

st.title("💄 Lipstick QnA Chatbot")
st.write("Ask questions about a specific lipstick. Follow up without repeating the name — the app will carry over context.")

# Sidebar: optional debug toggle
with st.sidebar:
    st.checkbox("Show debug info", value=st.session_state.show_debug, key="show_debug")
    if st.session_state.show_debug:
        with st.expander("Configuration", expanded=False):
            cfg = st.session_state.cfg or {}
            st.write({
                "model": cfg.get("model"),
                "system_prompt_path": cfg.get("system_prompt_path"),
                "embedding_model": cfg.get("embedding_model"),
                "pinecone_index": cfg.get("pc_index_name"),
                "namespace": cfg.get("pc_namespace"),
                "top_k": cfg.get("pc_top_k"),
                "max_tokens": cfg.get("max_tokens"),
                "temperature": cfg.get("temperature"),
            })
            st.caption(f"Session history file: {st.session_state.history_path}")

# Chat input & display
user_query = st.chat_input("Type your question…")

# Render history
for msg in st.session_state.conv_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.spinner("Thinking…"):
        answer = do_rag_and_respond(user_query)
    with st.chat_message("assistant"):
        st.markdown(answer)
