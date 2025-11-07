#!/usr/bin/env python3
"""
Anthropic CLI Chatbot with Improved Product Context Tracking

Usage:
  - Ensure your .env contains ANTHROPIC_API_KEY.
  - Optionally set ANTHROPIC_MODEL to choose a specific model.
    Examples:
      ANTHROPIC_MODEL=claude-4.1-opus
      ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
      ANTHROPIC_MODEL=claude-3-opus-20240229

Run:
  python chatbot.py
"""

import os
import sys
import shutil
import json  # ADDED MISSING IMPORT
from typing import List, Dict, Any
import logging
import time

from dotenv import load_dotenv
from anthropic import Anthropic, AnthropicError
from openai import OpenAI
from pinecone import Pinecone


def load_config():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set in your environment or .env file.")
        sys.exit(1)

    # Default to Opus; user requested "anthropic 4.1 opus" — allow override via env
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # System prompt: ONLY from file. No inline env default.
    system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH")
    system_prompt = None
    used_prompt_path = None

    # If a path is provided and exists, load it; otherwise try well-known default in project root
    candidates = []
    if system_prompt_path:
        candidates.append(system_prompt_path)
    candidates.append(os.path.join(os.getcwd(), "lipstick_qa_system_message (1).txt"))

    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    system_prompt = f.read().strip()
                    used_prompt_path = path
                    break
            except Exception as e:
                print(f"[Warning] Failed to read prompt file '{path}': {e}")

    if system_prompt is None:
        print("Error: No system prompt file found. Set SYSTEM_PROMPT_PATH to a valid file or place 'lipstick_qa_system_message (1).txt' in the project root.")
        sys.exit(1)

    max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "2000"))
    temperature = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7"))

    # OpenAI for embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # Pinecone retrieval settings
    pc_api_key = os.getenv("PINECONE_API_KEY")
    pc_env = os.getenv("PINECONE_ENVIRONMENT")
    pc_index_name = os.getenv("PINECONE_INDEX_NAME")
    pc_namespace = os.getenv("PINECONE_NAMESPACE", "default")
    pc_top_k = int(os.getenv("PINECONE_TOP_K", "1"))

    return {
        "api_key": api_key,
        "model": model,
        "system": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "openai_api_key": openai_api_key,
        "embedding_model": embedding_model,
        "pc_api_key": pc_api_key,
        "pc_env": pc_env,
        "pc_index_name": pc_index_name,
        "pc_namespace": pc_namespace,
        "pc_top_k": pc_top_k,
        "system_prompt_path": used_prompt_path,
    }


def build_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


def build_openai_client(api_key):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def build_pinecone(cfg: Dict[str, Any]) -> Any:
    if not (cfg.get("pc_api_key") and cfg.get("pc_index_name")):
        logging.warning("Pinecone not configured: missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")
        return None
    pc = Pinecone(api_key=cfg["pc_api_key"])
    # Environment is auto-resolved in v5; kept for compatibility if needed
    try:
        index = pc.Index(cfg["pc_index_name"])
    except Exception as e:
        logging.error("Error initializing Pinecone index '%s': %s", cfg.get("pc_index_name"), e)
        return None
    logging.info("Using Pinecone index='%s', namespace='%s'", cfg.get("pc_index_name"), cfg.get("pc_namespace"))
    return index


def embed_query(client: OpenAI, model: str, text: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding



def query_mentions_previous(user_query: str, last_ref: dict) -> bool:
    """Return True if the current query already mentions the last referenced product/brand/shade/SKU.
    This avoids hardcoded brand lists by dynamically checking tokens from last_ref.
    """
    if not last_ref:
        return False
    lowered = user_query.lower()
    candidate_tokens: list[str] = []
    for key in ("Product", "Brand", "Shade", "SKU", "Product Name", "Product Line"):
        v = last_ref.get(key) or last_ref.get(key.lower())
        if not isinstance(v, str) or not v:
            continue
        # consider the full string and its word parts
        candidate_tokens.append(v)
        candidate_tokens.extend(v.replace("-", " ").split())
    # keep tokens with reasonable length to reduce false positives
    candidate_tokens = [t for t in candidate_tokens if isinstance(t, str) and len(t) >= 3]
    for t in candidate_tokens:
        if t.lower() in lowered:
            return True
    return False


# ----------------------
# Identity extraction from Pinecone metadata/content
# ----------------------
def parse_header_kv(text: str) -> dict:
    """Parse header like "[ SKU=...; Brand=...; Product=...; Shade=...; Product Line=...]" from a content string."""
    out: dict = {}
    if not isinstance(text, str):
        return out
    # find bracketed header
    start = text.find("[")
    end = text.find("]", start + 1)
    header = text[start + 1:end] if start != -1 and end != -1 else None
    if not header:
        return out
    # split by ';'
    parts = [p.strip() for p in header.split(";") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
                out[k.lower()] = v  # also store lowercase alias
    return out


def extract_identity_fields(md: dict) -> dict:
    """Return a dict with canonical keys Product, Brand, Shade, SKU, Product Line, Product Name from metadata.
    Handles lowercase variants and parses 'content' header if present.
    """
    ident: dict = {}
    if not isinstance(md, dict):
        return ident
    # direct fields (any casing variants)
    candidates = {
        "Product": ["Product", "product", "product_name", "name"],
        "Brand": ["Brand", "brand"],
        "Shade": ["Shade", "shade", "shade_name"],
        "SKU": ["SKU", "sku", "id", "product_id"],
        "Product Line": ["Product Line", "product_line", "line"],
        "Product Name": ["Product Name", "product_name"],
    }
    for canon, keys in candidates.items():
        for k in keys:
            if k in md and isinstance(md[k], str) and md[k].strip():
                ident[canon] = md[k].strip()
                break
    # parse header in content for missing ones
    header = parse_header_kv(md.get("content")) if "content" in md else {}
    for canon in ("Product", "Brand", "Shade", "SKU", "Product Line", "Product Name"):
        if canon not in ident:
            v = header.get(canon) or header.get(canon.lower())
            if isinstance(v, str) and v.strip():
                ident[canon] = v.strip()
    return ident


    

def retrieve_context(index: Any, vector: List[float], namespace: str, top_k: int, metadata_filter: Dict[str, Any] = None) -> tuple:
    t0 = time.perf_counter()
    kwargs = {
        "vector": vector,
        "top_k": top_k,
        "include_values": False,
        "include_metadata": True,
        "namespace": namespace,
    }
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    res = index.query(**kwargs)
    t1 = time.perf_counter()
    # For pinecone-client v5, result format has 'matches'
    matches = getattr(res, "matches", []) or res.get("matches", [])
    try:
        scores = [getattr(m, "score", None) if not isinstance(m, dict) else m.get("score") for m in matches]
        logging.info(
            "Pinecone retrieved %d matches (top_k=%d) in %.3fs. Scores=%s%s",
            len(matches), top_k, (t1 - t0), scores,
            f", filter={metadata_filter}" if metadata_filter else "",
        )
    except Exception:
        logging.info(
            "Pinecone retrieved %d matches (top_k=%d) in %.3fs.%s",
            len(matches), top_k, (t1 - t0),
            f" filter={metadata_filter}" if metadata_filter else "",
        )
    return matches, (t1 - t0)


def extract_text_from_metadata(md: Dict[str, Any]) -> str:
    # Try common keys used in RAG pipelines
    for key in ("text", "content", "chunk", "page_content", "body"):
        if md and key in md and isinstance(md[key], str):
            return md[key]
    # Fallback: join all string values
    if md:
        parts = [str(v) for v in md.values() if isinstance(v, str)]
        if parts:
            return "\n".join(parts)
    return ""


def get_terminal_width(default: int = 100) -> int:
    try:
        width = shutil.get_terminal_size().columns
        return max(60, min(width, 160))
    except Exception:
        return default


def render_assistant(text: str) -> None:
    width = get_terminal_width()
    print("\n" + "=" * width)
    print("Assistant:\n")
    print(text.strip())
    print("=" * width + "\n")


def setup_logging() -> None:
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))


# ----------------------
# Persistent conversation history (JSONL)
# ----------------------
def load_conv_history(path: str, max_turns: int) -> List[Dict[str, str]]:
    """Load last max_turns user+assistant pairs from a JSONL file as Anthropic messages.
    Each JSONL line should be an object: {"role": "user"|"assistant", "content": str}.
    Returns a list of messages usable directly in Anthropic's messages.create.
    """
    history: List[Dict[str, str]] = []
    if not path or not os.path.isfile(path):
        return history
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # Keep only the last 2*max_turns lines (user+assistant per turn)
        if max_turns > 0:
            lines = lines[-(max_turns * 2):]
        for ln in lines:
            try:
                obj = json.loads(ln)
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    history.append({"role": role, "content": content})
            except Exception:
                continue
    except Exception:
        pass
    return history


def append_conv_history(path: str, role: str, content: str) -> None:
    """Append a single message to the JSONL history file, creating directories as needed."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")
    except Exception:
        # Non-fatal: don't block the chat loop
        logging.debug("Failed to append to history file: %s", path)


def main():
    setup_logging()
    logger = logging.getLogger("chatbot")
    cfg = load_config()
    client = build_client(cfg["api_key"])
    oai = build_openai_client(cfg.get("openai_api_key"))
    pc_index = build_pinecone(cfg)

    print("Anthropic CLI Chatbot")
    print(f"Model: {cfg['model']}")
    if cfg.get("system_prompt_path"):
        print(f"System prompt file: {cfg['system_prompt_path']}")
    print("Type 'exit' or 'quit' to end. Press Ctrl+C to abort.\n")

    # Persistent conversation history path
    conv_history_path = os.getenv("CONV_HISTORY_PATH", os.path.join("history", "conversation.jsonl"))

    # Log startup configuration (without secrets)
    logger.info(
        "Startup config: model=%s, prompt_path=%s, embedding_model=%s, pinecone_index=%s, namespace=%s, top_k=%s, max_tokens=%s, temperature=%s",
        cfg.get("model"), cfg.get("system_prompt_path"), cfg.get("embedding_model"),
        cfg.get("pc_index_name"), cfg.get("pc_namespace"), cfg.get("pc_top_k"),
        cfg.get("max_tokens"), cfg.get("temperature"),
    )
    logger.info("Conversation history file: %s", conv_history_path)

    # In-memory conversation history so the LLM knows prior turns (per session)
    # Each entry: {"role": "user"|"assistant", "content": str}
    max_turns = int(os.getenv("HISTORY_MAX_TURNS", "5"))  # number of prior user+assistant turns to keep
    # Seed from persistent history
    conv_history: List[Dict[str, str]] = load_conv_history(conv_history_path, max_turns)

    # Track last resolved product context from retrieval (for follow-up questions)
    last_ref: Dict[str, Any] = {}

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # RAG: embed query -> retrieve from Pinecone -> build strict-context prompt
            turn_t0 = time.perf_counter()
            context_blocks: List[str] = []
            if oai and pc_index:
                try:
                    t_emb0 = time.perf_counter()
                    # If previous turn identified a product/shade and user didn't name it now, carry over
                    metadata_filter = None
                    effective_query = user_input
                    
                    # Check if we have previous product context
                    prev_product = (last_ref.get("Product") or last_ref.get("product") or "").strip()
                    prev_brand = (last_ref.get("Brand") or last_ref.get("brand") or "").strip()
                    prev_shade = (last_ref.get("Shade") or last_ref.get("shade") or "").strip()
                    prev_sku = (last_ref.get("SKU") or last_ref.get("sku") or "").strip()
                    
                    # Detect if user already mentioned the previous product context (dynamic, no hardcoding)
                    mentions_product = query_mentions_previous(user_input, last_ref)
                    
                    # If we have previous context and user didn't mention product, augment query
                    if (prev_product or prev_brand or prev_sku) and not mentions_product:
                        # Augment the query with previous product context
                        steer_bits = []
                        # Prefer product; add brand only if not already part of product text
                        if prev_product:
                            steer_bits.append(prev_product)
                            if prev_brand and prev_brand.lower() not in prev_product.lower():
                                steer_bits.append(prev_brand)
                        else:
                            if prev_brand:
                                steer_bits.append(prev_brand)
                            if prev_shade:
                                steer_bits.append(prev_shade)
                        # Make unique while preserving order
                        seen = set()
                        steer_bits = [b for b in steer_bits if not (b in seen or seen.add(b))]
                        
                        if steer_bits:
                            product_context = " ".join(steer_bits)
                            # Prepend product context to query for better semantic search
                            # Include SKU token in the text (helps embeddings) if available
                            if prev_sku:
                                effective_query = f"{product_context} [SKU: {prev_sku}] {user_input}"
                            else:
                                effective_query = f"{product_context} {user_input}"
                            logger.info("🔗 Augmented query with previous product: '%s'", product_context)
                            # No Pinecone filter — we rerank client-side by SKU
                            metadata_filter = None
                            logger.debug("Using client-side rerank by SKU (no metadata filter)")

                    vec = embed_query(oai, cfg["embedding_model"], effective_query)
                    t_emb1 = time.perf_counter()
                    logger.info("Embedding time: %.3fs (model=%s, dim=%d)", (t_emb1 - t_emb0), cfg["embedding_model"], len(vec))
                    logger.debug("Embedded query into vector of length %d", len(vec))
                    matches, _ = retrieve_context(pc_index, vec, cfg["pc_namespace"], cfg["pc_top_k"], metadata_filter=None)
                    # Optional: rerank matches to prioritize items with matching carried SKU
                    if matches and last_ref.get("SKU"):
                        prev_sku_val = last_ref.get("SKU")
                        def sku_match_first(item):
                            md_i = getattr(item, "metadata", None) or item.get("metadata", {})
                            ident_i = extract_identity_fields(md_i)
                            id_i = getattr(item, "id", None) or item.get("id")
                            sku_equal = (ident_i.get("SKU") == prev_sku_val)
                            id_contains = (isinstance(id_i, str) and prev_sku_val in id_i)
                            # Primary key: 0 for positive match, 1 otherwise
                            primary = 0 if (sku_equal or id_contains) else 1
                            # Secondary: negative score for descending sort if available
                            score = getattr(item, "score", None)
                            if score is None and isinstance(item, dict):
                                score = item.get("score")
                            secondary = -(score or 0.0)
                            return (primary, secondary)
                        try:
                            matches.sort(key=sku_match_first)
                            logger.debug("Reranked matches to prioritize SKU/id match for SKU=%s", prev_sku_val)
                        except Exception:
                            pass

                    for m in matches:
                        md = getattr(m, "metadata", None) or m.get("metadata", {})
                        txt = extract_text_from_metadata(md)
                        if txt:
                            context_blocks.append(txt)
                    
                    # Update last reference with best match's identity fields
                    if matches:
                        top_md = getattr(matches[0], "metadata", None) or matches[0].get("metadata", {})
                        # Extract and store product details for next turn (robust to lowercase and header-in-content)
                        ident = extract_identity_fields(top_md)
                        if ident:
                            last_ref.update(ident)
                        # Log stored context (helpful for debugging)
                        stored_info = {k: v for k, v in last_ref.items() if k in ("Product", "Brand", "Shade", "SKU")}
                        if stored_info:
                            logger.info("💾 Stored product context for next turn: %s", stored_info)
                            
                except Exception as e:
                    logger.warning("RAG retrieval failed: %s", e)

            t_ctx0 = time.perf_counter()
            context_text = "\n\n---\n\n".join(context_blocks).strip()
            logger.debug("Context length (chars)=%d across %d blocks", len(context_text), len(context_blocks))
            t_ctx1 = time.perf_counter()
            logger.info("RAG assembly time: %.3fs", (t_ctx1 - t_ctx0))

            # Build messages: only pass retrieved context + question
            # If no context, instruct model to say it doesn't know
            rag_instructions = (
                "You are a precise assistant. Answer ONLY using the provided context. "
                "If the answer isn't in the context, say 'I don't know' succinctly."
            )
            question_block = f"Question:\n{user_input}"
            if context_text:
                user_payload = f"{question_block}\n\nContext:\n{context_text}"
            else:
                user_payload = f"{question_block}\n\nContext:\n[No relevant context retrieved]"

            # Include conversation history window in messages
            history_window = conv_history[-(max_turns * 2):] if max_turns > 0 else []
            # Save current user turn to memory and persistent file
            conv_history.append({"role": "user", "content": user_payload})
            append_conv_history(conv_history_path, "user", user_payload)

            try:
                t_llm0 = time.perf_counter()
                response = client.messages.create(
                    model=cfg["model"],
                    max_tokens=cfg["max_tokens"],
                    temperature=cfg["temperature"],
                    system=f"{cfg['system']}\n\n{rag_instructions}",
                    messages=history_window + [{"role": "user", "content": user_payload}],
                )
                t_llm1 = time.perf_counter()
                logger.info("LLM latency: %.3fs (model=%s)", (t_llm1 - t_llm0), cfg["model"])
            except AnthropicError as e:
                logging.error("Anthropic API error: %s", e)
                continue

            # Extract text content from the response
            # response.content is a list of content blocks; we join text blocks
            parts: List[str] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", ""))
            assistant_text = "\n".join(p for p in parts if p)

            if assistant_text:
                render_assistant(assistant_text)
                # Save assistant reply to history (memory + persistent file)
                conv_history.append({"role": "assistant", "content": assistant_text})
                append_conv_history(conv_history_path, "assistant", assistant_text)
            else:
                logging.warning("Empty response from model.")

            turn_t1 = time.perf_counter()
            logger.info("Turn total time: %.3fs", (turn_t1 - turn_t0))


    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")


if __name__ == "__main__":
    main()