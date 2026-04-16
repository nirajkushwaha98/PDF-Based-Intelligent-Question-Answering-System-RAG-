from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import PyPDF2
import os
import re
import math
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

resolved_model_name = None

# Simple in-memory storage (for production use database)
pdf_contents = {}
conversation_history = {}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "or", "if", "you", "your", "about", "into",
    "can", "could", "should", "would", "what", "which", "when", "where", "how", "why",
}

COMMON_JOINED_FIXES = {
    "areused": "are used",
    "isused": "is used",
    "toshow": "to show",
    "tocompare": "to compare",
    "toexplain": "to explain",
    "forexample": "for example",
    "adistribution": "a distribution",
    "abarchart": "a bar chart",
    "areuseful": "are useful",
    "invery": "in very",
    "scatterplots": "scatter plots",
}


def normalize_extracted_text(text):
    """Clean noisy PDF-extracted text for better retrieval/prompt quality."""
    cleaned = (text or "").replace("\r", "\n").replace("\u00a0", " ")
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)
    cleaned = re.sub(r"[•●▪◦]", " ", cleaned)
    cleaned = re.sub(r"\s+-\s+", " ", cleaned)
    cleaned = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([,:;])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([a-z])\s+([A-Z][A-Za-z ]{2,25}:\s)", r"\1. \2", cleaned)
    for bad, good in COMMON_JOINED_FIXES.items():
        cleaned = re.sub(rf"\b{bad}\b", good, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def extract_pdf_text(file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        metadata = {
            "total_pages": len(pdf_reader.pages),
            "file_name": file.filename,
            "upload_time": datetime.now().isoformat(),
        }

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n--- Page {page_num + 1} ---\n"
            text += normalize_extracted_text(page_text)

        return text, metadata
    except Exception as e:
        return None, {"error": str(e)}


def chunk_text(text, chunk_size=2000, overlap=200):
    """Split text into chunks with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def tokenize(text):
    words = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return [w for w in words if w not in STOP_WORDS]


def find_relevant_chunks(query, pdf_text, num_chunks=3):
    """Simple keyword-based retrieval."""
    chunks = chunk_text(pdf_text)
    query_words = set(tokenize(query))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(tokenize(chunk))
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored_chunks.append((overlap, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    relevant_chunks = [chunk for _, chunk in scored_chunks[:num_chunks]]
    return relevant_chunks if relevant_chunks else chunks[:num_chunks]


def build_extractive_fallback_answer(relevant_chunks):
    """Return a concise extractive fallback from retrieved context."""
    if not relevant_chunks:
        return "I couldn't find this information in the document."

    excerpts = []
    for index, chunk in enumerate(relevant_chunks[:3], start=1):
        compact = " ".join((chunk or "").split())
        if len(compact) > 320:
            compact = compact[:320].rsplit(" ", 1)[0] + "..."
        excerpts.append(f"{index}. {compact}")

    return "I couldn't find an exact answer, but related content says:\n" + "\n".join(excerpts)


def get_gemini_client():
    """Create Gemini client only when needed."""
    keys = get_configured_api_keys()
    if not keys:
        return None
    return genai.Client(api_key=keys[0])


def get_configured_api_keys():
    """Return unique API keys in priority order."""
    keys = []

    primary = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if primary:
        keys.append(primary)

    extra = (os.getenv("GEMINI_API_KEYS") or "").strip()
    if extra:
        keys.extend([item.strip() for item in extra.split(",") if item.strip()])

    unique_keys = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def get_gemini_clients():
    """Create Gemini clients for all configured keys."""
    return [genai.Client(api_key=key) for key in get_configured_api_keys()]


def resolve_gemini_model_name(gemini_client):
    """Pick an available model for generate_content."""
    global resolved_model_name
    if resolved_model_name:
        return resolved_model_name

    requested = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
    preferred = [
        requested,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
    ]

    try:
        available = []
        for model in gemini_client.models.list():
            actions = set(model.supported_actions or [])
            if "generateContent" not in actions:
                continue
            model_name = (model.name or "").replace("models/", "")
            if model_name:
                available.append(model_name)

        for candidate in preferred:
            if candidate in available:
                resolved_model_name = candidate
                return resolved_model_name

        for candidate in preferred:
            match = next((name for name in available if name.startswith(f"{candidate}-")), None)
            if match:
                resolved_model_name = match
                return resolved_model_name

        flash_match = next((name for name in available if "flash" in name), None)
        if flash_match:
            resolved_model_name = flash_match
            return resolved_model_name
    except Exception:
        pass

    resolved_model_name = requested
    return resolved_model_name


def generate_with_fallback(gemini_client, prompt, max_output_tokens=1024):
    """Generate content with model fallback on NOT_FOUND."""
    primary_model = resolve_gemini_model_name(gemini_client)
    retry_models = [
        primary_model,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
    ]

    last_error = None
    seen = set()
    for model_name in retry_models:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        try:
            return gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"max_output_tokens": max_output_tokens},
            )
        except Exception as api_error:
            last_error = api_error
            message = str(api_error).lower()
            retryable = (
                "not_found" in message
                or "not found" in message
                or "resource_exhausted" in message
                or "quota exceeded" in message
                or "rate limit" in message
            )
            if not retryable:
                break

    raise last_error


def classify_gemini_error(api_error):
    """Map Gemini exceptions to stable API responses."""
    raw_error = str(api_error)
    lower_error = raw_error.lower()

    if (
        "resource_exhausted" in lower_error
        or "quota exceeded" in lower_error
        or "rate limit" in lower_error
    ):
        retry_seconds = None
        retry_in_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", lower_error)
        retry_delay_match = re.search(r"retrydelay[^0-9]*([0-9]+)s", lower_error)
        if retry_in_match:
            retry_seconds = math.ceil(float(retry_in_match.group(1)))
        elif retry_delay_match:
            retry_seconds = int(retry_delay_match.group(1))

        message = "Gemini API quota is exhausted for this key."
        if retry_seconds is not None:
            message += f" Retry after about {retry_seconds} seconds."
        message += " Details: https://ai.google.dev/gemini-api/docs/rate-limits"
        return {"status_code": 429, "reason": "quota_exhausted", "message": message}

    if "not_found" in lower_error and "model" in lower_error:
        return {
            "status_code": 400,
            "reason": "model_not_found",
            "message": "Configured Gemini model is unavailable. Set GEMINI_MODEL to a supported model.",
        }

    if (
        "permission_denied" in lower_error
        or "api key not valid" in lower_error
        or "invalid api key" in lower_error
    ):
        return {"status_code": 401, "reason": "invalid_api_key", "message": "Gemini API key is invalid or unauthorized."}

    if (
        "nodename nor servname provided" in lower_error
        or "failed to establish a new connection" in lower_error
        or "name or service not known" in lower_error
    ):
        return {"status_code": 503, "reason": "network_error", "message": "Network error while calling Gemini API. Check internet and retry."}

    return {"status_code": 502, "reason": "upstream_error", "message": f"Gemini API request failed: {raw_error}"}


def build_conversation_transcript(messages):
    lines = []
    for message in messages:
        role = "Assistant" if message.get("role") == "assistant" else "User"
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    key_count = len(get_configured_api_keys())
    return jsonify({"status": "healthy", "mode": "api", "ai_configured": key_count > 0, "key_count": key_count}), 200


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF upload."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        text, metadata = extract_pdf_text(file)
        if text is None:
            return jsonify(metadata), 400

        pdf_id = f"pdf_{len(pdf_contents) + 1}_{int(datetime.now().timestamp())}"
        pdf_contents[pdf_id] = {
            "text": text,
            "metadata": metadata,
            "chunks": chunk_text(text),
        }
        conversation_history[pdf_id] = []

        return jsonify({
            "pdf_id": pdf_id,
            "status": "uploaded",
            "pages": metadata["total_pages"],
            "filename": metadata["file_name"],
            "message": f"PDF uploaded successfully with {metadata['total_pages']} pages",
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer questions using Gemini API."""
    try:
        data = request.get_json(silent=True) or {}
        pdf_id = data.get('pdf_id')
        question = (data.get('question') or '').strip()

        if not pdf_id or not question:
            return jsonify({"error": "pdf_id and question required"}), 400
        if pdf_id not in pdf_contents:
            return jsonify({"error": "PDF not found"}), 404

        gemini_clients = get_gemini_clients()
        if not gemini_clients:
            return jsonify({
                "error": "GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured. Add it to .env and restart backend."
            }), 503

        pdf_data = pdf_contents[pdf_id]
        relevant_chunks = find_relevant_chunks(question, pdf_data["text"], num_chunks=5)
        context = "\n\n".join(relevant_chunks)

        conversation_history.setdefault(pdf_id, [])
        transcript = build_conversation_transcript(conversation_history[pdf_id])

        prompt = f"""You are a helpful assistant that answers questions about a PDF document.
Use only the provided document context.
If the answer is not explicitly present, provide the closest relevant information from the context and mention uncertainty.
Write a concise, clear answer in plain English.

Conversation History:
{transcript if transcript else "No previous conversation."}

Document Context:
{context}

User Question:
{question}"""

        response = None
        mapped = None
        used_client = None
        for idx, gemini_client in enumerate(gemini_clients, start=1):
            try:
                response = generate_with_fallback(gemini_client, prompt, max_output_tokens=1024)
                used_client = gemini_client
                break
            except Exception as api_error:
                mapped = classify_gemini_error(api_error)
                can_try_next_key = idx < len(gemini_clients)
                if can_try_next_key and mapped["reason"] in {"quota_exhausted", "invalid_api_key"}:
                    continue
                break

        if response is None:
            if mapped and mapped["reason"] == "quota_exhausted" and len(gemini_clients) > 1:
                mapped["message"] = (
                    "Gemini API quota is exhausted for all configured keys. "
                    "Add another key in GEMINI_API_KEYS or wait for quota reset. "
                    "Details: https://ai.google.dev/gemini-api/docs/rate-limits"
                )
            mapped = mapped or {"status_code": 502, "message": "Gemini API request failed."}
            return jsonify({"error": mapped["message"]}), mapped["status_code"]

        answer = ""
        try:
            answer = (response.text or "").strip()
        except Exception:
            answer = ""

        if not answer:
            parts = []
            candidates = getattr(response, "candidates", []) or []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            answer = "\n".join(parts).strip()

        if not answer:
            answer = "I couldn't generate a response from the model."

        # Avoid overly strict "not found" responses when relevant context exists.
        if "couldn't find this information in the document" in answer.lower() and relevant_chunks:
            answer = build_extractive_fallback_answer(relevant_chunks)

        conversation_history[pdf_id].append({"role": "user", "content": question})
        conversation_history[pdf_id].append({"role": "assistant", "content": answer})

        return jsonify({
            "answer": answer,
            "pdf_id": pdf_id,
            "context_used": len(relevant_chunks),
            "status": "success",
            "mode": "api",
            "model": resolve_gemini_model_name(used_client),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/pdfs', methods=['GET'])
def list_pdfs():
    """List all uploaded PDFs."""
    pdfs = []
    for pdf_id, data in pdf_contents.items():
        pdfs.append({
            "pdf_id": pdf_id,
            "filename": data["metadata"]["file_name"],
            "pages": data["metadata"]["total_pages"],
            "upload_time": data["metadata"]["upload_time"],
        })
    return jsonify({"pdfs": pdfs}), 200


@app.route('/clear/<pdf_id>', methods=['DELETE'])
def clear_pdf(pdf_id):
    """Clear a specific PDF and its history."""
    if pdf_id in pdf_contents:
        del pdf_contents[pdf_id]
        if pdf_id in conversation_history:
            del conversation_history[pdf_id]
        return jsonify({"status": "cleared"}), 200
    return jsonify({"error": "PDF not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
