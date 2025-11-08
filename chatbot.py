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
    pc_top_k = int(os.getenv("PINECONE_TOP_K", "3"))

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


def reformulate_query_with_llm(client: Anthropic, user_query: str, conv_history: List[Dict[str, str]], model: str = "claude-haiku-4-5-20251001") -> str:
    """Use LLM to reformulate user query based on conversation history.
    This replaces hardcoded pattern matching with intelligent context understanding.
    
    Args:
        client: Anthropic client
        user_query: Current user query
        conv_history: List of previous messages [{"role": "user"|"assistant", "content": str}]
        model: Model to use for reformulation (default: fast Haiku model)
    
    Returns:
        Reformulated query optimized for semantic search
    """
    # Take only the last 2-3 turns for context (avoid token bloat)
    recent_history = conv_history[-(4):] if len(conv_history) > 4 else conv_history
    
    # Build conversation context string
    context_lines = []
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            # Extract just the question part (remove "Question:" and "Context:" sections if present)
            if "Question:" in content:
                question_part = content.split("Question:")[1].split("Context:")[0].strip()
                context_lines.append(f"User: {question_part}")
            else:
                context_lines.append(f"User: {content[:200]}")  # Limit length
        elif role == "assistant":
            context_lines.append(f"Assistant: {content[:200]}")  # Limit length
    
    conversation_context = "\n".join(context_lines) if context_lines else "[No previous conversation]"
    
    reformulation_prompt = f"""You are a query reformulation assistant for a product Q&A system.

Your task: Check if the conversation history has enough information to answer the user's question. If yes, answer directly. If no, generate an optimized search query. If the question is ambiguous, ask for clarification.

Rules:
1. First check: Does the conversation history contain enough information to answer this question?
   - If YES: Return the answer directly based on the history (prefix with "ANSWER: ")
   - If NO: Continue to step 2

2. Ambiguity check:
   - If the user refers to "this lipstick", "this product", "it", "this", etc. WITHOUT naming a specific product:
     * Check conversation history for [Retrieved: ...] markers with product context
     * If NO clear product context exists in recent history: Return "CLARIFY: Could you please specify which lipstick you're asking about? (Include the brand and product name)"
     * If product context exists but you're uncertain which product they mean: Return "CLARIFY: I see multiple products mentioned. Which one are you asking about - [list products]?"
   - If the question is clear and specific, proceed to step 3

3. For search query generation:
   - IMPORTANT: Always include BOTH the product identifier AND the key question terms
   - Example: "What is the shade of MAC Ruby Woo?" → "MAC Ruby Woo shade" (NOT just "MAC Ruby Woo")
   - Example: "Does it last long?" → "[Product Name] longevity lasting" (NOT just "[Product Name]")
   - Look for [Retrieved: ...] markers in history - they contain Product, Brand, and SKU from previous queries
   - If the question is a follow-up (e.g., "what is the price?", "does it last long?"), include the product name AND the question attribute (price, longevity, etc.)
   - SKU with the full product Name also is the most precise identifier 
   - If the question mentions a NEW product name, use ONLY that new product (ignore previous context)
   - Keep the query concise but include: product identifier + key question terms (shade, price, finish, longevity, ingredients, etc.)
   - Remove filler words like "tell me", "I want to know", "what is", etc.
   - Preserve important qualifiers like shade names/numbers, skin tone, skin concerns, etc.

Conversation History:
{conversation_context}

Current Question: {user_query}

Response (either "ANSWER: <answer>", "CLARIFY: <clarification question>", or "<search_query>"):"""
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,  # Increased for direct answers from history
            temperature=0.0,  # Deterministic
            messages=[{"role": "user", "content": reformulation_prompt}]
        )
        
        # Extract response
        reformulated = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                reformulated = getattr(block, "text", "").strip()
                break
        
        if reformulated and len(reformulated) > 3:
            # Check if LLM answered directly from history
            if reformulated.startswith("ANSWER:"):
                answer = reformulated[7:].strip()  # Remove "ANSWER: " prefix
                logging.info("⚡ Answering from conversation history (no RAG needed)")
                return f"ANSWER:{answer}"  # Return with marker for main loop
            # Check if LLM needs clarification
            elif reformulated.startswith("CLARIFY:"):
                clarification = reformulated[8:].strip()  # Remove "CLARIFY: " prefix
                logging.info("❓ Asking for clarification")
                return f"CLARIFY:{clarification}"  # Return with marker for main loop
            # Otherwise it's a search query
            logging.info("🔄 Query reformulated: '%s' → '%s'", user_query, reformulated)
            return reformulated
        else:
            logging.warning("LLM reformulation returned empty, using original query")
            return user_query
            
    except Exception as e:
        logging.warning("Query reformulation failed: %s, using original query", e)
        return user_query




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
                    # Use LLM to reformulate query based on conversation history
                    # This intelligently handles follow-up questions and new product queries
                    effective_query = reformulate_query_with_llm(client, user_input, conv_history, cfg["model"])

                    # Check if LLM answered directly from history
                    if effective_query.startswith("ANSWER:"):
                        # Extract the direct answer and skip RAG + main LLM
                        direct_answer = effective_query[7:].strip()
                        logger.info("⚡ Using direct answer from conversation history")
                        render_assistant(direct_answer)
                        # Save to history
                        conv_history.append({"role": "user", "content": user_input})
                        append_conv_history(conv_history_path, "user", user_input)
                        conv_history.append({"role": "assistant", "content": direct_answer})
                        append_conv_history(conv_history_path, "assistant", direct_answer)
                        turn_t1 = time.perf_counter()
                        logger.info("Turn total time: %.3fs (answered from history)", (turn_t1 - turn_t0))
                        continue  # Skip to next user input
                    
                    # Check if LLM needs clarification
                    if effective_query.startswith("CLARIFY:"):
                        # Extract the clarification question and ask user
                        clarification_msg = effective_query[8:].strip()
                        logger.info("❓ Requesting clarification from user")
                        render_assistant(clarification_msg)
                        # Save to history
                        conv_history.append({"role": "user", "content": user_input})
                        append_conv_history(conv_history_path, "user", user_input)
                        conv_history.append({"role": "assistant", "content": clarification_msg})
                        append_conv_history(conv_history_path, "assistant", clarification_msg)
                        turn_t1 = time.perf_counter()
                        logger.info("Turn total time: %.3fs (asked for clarification)", (turn_t1 - turn_t0))
                        continue  # Skip to next user input
                    
                    # Otherwise perform RAG retrieval
                    vec = embed_query(oai, cfg["embedding_model"], effective_query)
                    t_emb1 = time.perf_counter()
                    logger.info("Embedding time: %.3fs (model=%s, dim=%d)", (t_emb1 - t_emb0), cfg["embedding_model"], len(vec))
                    logger.debug("Embedded query into vector of length %d", len(vec))
                    matches, _ = retrieve_context(pc_index, vec, cfg["pc_namespace"], cfg["pc_top_k"], metadata_filter=None)
                    
                    # Extract SKU and product info from top match for context tracking
                    retrieved_sku = None
                    retrieved_product = None
                    retrieved_brand = None
                    if matches:
                        top_match_md = getattr(matches[0], "metadata", None) or matches[0].get("metadata", {})
                        # Try multiple field names for SKU
                        retrieved_sku = (top_match_md.get("SKU") or top_match_md.get("sku") or 
                                        top_match_md.get("id") or top_match_md.get("product_id") or "").strip()
                        # Try multiple field names for product/brand
                        retrieved_product = (top_match_md.get("Product") or top_match_md.get("product") or 
                                           top_match_md.get("product_name") or top_match_md.get("name") or "").strip()
                        retrieved_brand = (top_match_md.get("Brand") or top_match_md.get("brand") or "").strip()
                        
                        if retrieved_sku or retrieved_product:
                            logger.info("📦 Retrieved product context - SKU: %s, Product: %s, Brand: %s", 
                                      retrieved_sku or "N/A", retrieved_product or "N/A", retrieved_brand or "N/A")
                    
                    for m in matches:
                        md = getattr(m, "metadata", None) or m.get("metadata", {})
                        txt = extract_text_from_metadata(md)
                        if txt:
                            context_blocks.append(txt)
                            
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
            # Build user payload with product metadata for context tracking
            question_block = f"Question:\n{user_input}"
            
            # Add SKU/product metadata to history for reformulation context
            product_context_line = ""
            if 'retrieved_sku' in locals() and (retrieved_sku or retrieved_product):
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