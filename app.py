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
    retrieve_context,
    extract_text_from_metadata,
    extract_identity_fields,
    query_mentions_previous,
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
    if "last_ref" not in st.session_state:
        st.session_state.last_ref = {}
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

    # Build context via RAG similar to CLI flow
    context_blocks: List[str] = []
    effective_query = user_input

    last_ref: Dict[str, Any] = st.session_state.last_ref
    prev_product = (last_ref.get("Product") or last_ref.get("product") or "").strip()
    prev_brand = (last_ref.get("Brand") or last_ref.get("brand") or "").strip()
    prev_shade = (last_ref.get("Shade") or last_ref.get("shade") or "").strip()
    prev_sku = (last_ref.get("SKU") or last_ref.get("sku") or "").strip()

    mentions_product = query_mentions_previous(user_input, last_ref)

    if (prev_product or prev_brand or prev_sku) and not mentions_product:
        steer_bits: List[str] = []
        if prev_product:
            steer_bits.append(prev_product)
            if prev_brand and prev_brand.lower() not in prev_product.lower():
                steer_bits.append(prev_brand)
        else:
            if prev_brand:
                steer_bits.append(prev_brand)
            if prev_shade:
                steer_bits.append(prev_shade)
        seen = set()
        steer_bits = [b for b in steer_bits if not (b in seen or seen.add(b))]
        if steer_bits:
            product_context = " ".join(steer_bits)
            if prev_sku:
                effective_query = f"{product_context} [SKU: {prev_sku}] {user_input}"
            else:
                effective_query = f"{product_context} {user_input}"
            if st.session_state.show_debug:
                st.info(f"Using previous product context: {product_context}")

    # Embeddings + retrieve
    if oai and pc_index:
        try:
            vec = embed_query(oai, cfg["embedding_model"], effective_query)
            matches, _ = retrieve_context(pc_index, vec, cfg["pc_namespace"], cfg["pc_top_k"], metadata_filter=None)
            # Optional rerank by carried SKU or SKU-in-id
            if matches and last_ref.get("SKU"):
                prev_sku_val = last_ref.get("SKU")
                def sku_match_first(item):
                    md_i = getattr(item, "metadata", None) or item.get("metadata", {})
                    ident_i = extract_identity_fields(md_i)
                    id_i = getattr(item, "id", None) or item.get("id")
                    sku_equal = (ident_i.get("SKU") == prev_sku_val)
                    id_contains = (isinstance(id_i, str) and prev_sku_val in id_i)
                    primary = 0 if (sku_equal or id_contains) else 1
                    score = getattr(item, "score", None)
                    if score is None and isinstance(item, dict):
                        score = item.get("score")
                    secondary = -(score or 0.0)
                    return (primary, secondary)
                try:
                    matches.sort(key=sku_match_first)
                except Exception:
                    pass

            for m in matches:
                md = getattr(m, "metadata", None) or m.get("metadata", {})
                txt = extract_text_from_metadata(md)
                if txt:
                    context_blocks.append(txt)

            # Update last_ref from top match
            if matches:
                top_md = getattr(matches[0], "metadata", None) or matches[0].get("metadata", {})
                ident = extract_identity_fields(top_md)
                if ident:
                    st.session_state.last_ref.update(ident)
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
    if context_text:
        user_payload = f"{question_block}\n\nContext:\n{context_text}"
    else:
        user_payload = f"{question_block}\n\nContext:\n[No relevant context retrieved]"

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
