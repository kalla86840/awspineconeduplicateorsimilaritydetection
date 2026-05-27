import json
import os
import re
import time
from collections import Counter
from pathlib import Path

KNOWLEDGE_PATH = Path(os.environ.get("RAG_KNOWLEDGE_PATH", "knowledge/open_ai_rag_knowledge.txt"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY_SECRET_ARN = os.environ.get("OPENAI_API_KEY_SECRET_ARN")
PINECONE_API_KEY_SECRET_ARN = os.environ.get("PINECONE_API_KEY_SECRET_ARN")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "news-demo")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "news")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION = int(os.environ.get("PINECONE_DIMENSION", "1536"))
PINECONE_METRIC = os.environ.get("PINECONE_METRIC", "cosine")
ENABLE_PINECONE = os.environ.get("ENABLE_PINECONE", "true").lower() == "true"
PINECONE_UPSERT_ON_QUERY = os.environ.get("PINECONE_UPSERT_ON_QUERY", "false").lower() == "true"
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "700"))
TOP_K = int(os.environ.get("TOP_K", "3"))
_OPENAI_API_KEY = None
_PINECONE_API_KEY = None
_PINECONE_INDEXED = False

AGENTS = [
    {"name": "manual_retrieval_agent", "role": "Document retrieval analyst", "instructions": "You are Agent 1, the document retrieval analyst. Use only the retrieved manual sections to identify the sections that answer the question. Return practical findings and cite section titles. If the retrieved context is insufficient, say what is missing."},
    {"name": "procedure_agent", "role": "Step-by-step procedure specialist", "instructions": "You are Agent 2, the procedure specialist. Use only the retrieved manual sections to extract an ordered, user-safe procedure. Preserve important sequence, prerequisites, and checks from the document."},
    {"name": "safety_agent", "role": "Safety and escalation reviewer", "instructions": "You are Agent 3, the safety reviewer. Use only the retrieved manual sections to identify warnings, stop conditions, and when a qualified professional should be contacted."},
]

AGENT_OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"summary": {"type": "string"}, "findings": {"type": "array", "items": {"type": "string"}}, "citations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"id": {"type": "string"}, "title": {"type": "string"}}, "required": ["id", "title"]}}, "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}, "required": ["summary", "findings", "citations", "confidence"]}
FINAL_OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {"answer": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}}, "safety_notes": {"type": "array", "items": {"type": "string"}}, "citations": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {"id": {"type": "string"}, "title": {"type": "string"}}, "required": ["id", "title"]}}, "agent_consensus": {"type": "string"}}, "required": ["answer", "steps", "safety_notes", "citations", "agent_consensus"]}


def _tokenize(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    normalized = []
    for token in tokens:
        if token.startswith("connect"):
            normalized.append("connect")
        elif token.startswith("install"):
            normalized.append("install")
        elif token.startswith("leak"):
            normalized.append("leak")
        elif token.endswith("s") and len(token) > 3:
            normalized.append(token[:-1])
        else:
            normalized.append(token)
    return normalized


def load_documents(path=KNOWLEDGE_PATH):
    content = Path(path).read_text(encoding="utf-8").strip()
    sections = [section.strip() for section in re.split(r"\n\s*\n", content) if section.strip()]
    documents = []
    for index, section in enumerate(sections, start=1):
        lines = section.splitlines()
        title = lines[0].strip("# ").strip() if lines else f"Section {index}"
        documents.append({"id": f"doc-{index}", "title": title, "content": section})
    return documents


def retrieve(question, documents, top_k=TOP_K):
    question_terms = Counter(_tokenize(question))
    scored = []
    for document in documents:
        document_terms = Counter(_tokenize(f"{document['title']} {document['content']}"))
        score = sum(question_terms[term] * document_terms.get(term, 0) for term in question_terms)
        scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [document for score, document in scored if score > 0]
    if not matches:
        matches = [document for _, document in scored]
    return matches[:top_k]


def get_secret_value(secret_arn, env_var_name):
    value = os.environ.get(env_var_name)
    if value:
        return value
    if not secret_arn:
        raise ValueError(f"{env_var_name}_SECRET_ARN or {env_var_name} must be configured.")
    import boto3
    return boto3.client("secretsmanager").get_secret_value(SecretId=secret_arn)["SecretString"]


def get_openai_api_key():
    global _OPENAI_API_KEY
    if not _OPENAI_API_KEY:
        _OPENAI_API_KEY = get_secret_value(OPENAI_API_KEY_SECRET_ARN, "OPENAI_API_KEY")
    return _OPENAI_API_KEY


def get_pinecone_api_key():
    global _PINECONE_API_KEY
    if not _PINECONE_API_KEY:
        _PINECONE_API_KEY = get_secret_value(PINECONE_API_KEY_SECRET_ARN, "PINECONE_API_KEY")
    return _PINECONE_API_KEY


def embed_texts(client, texts):
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts, dimensions=PINECONE_DIMENSION)
    return [item.embedding for item in response.data]


def _list_pinecone_index_names(pc):
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return set(indexes.names())
    return {index["name"] for index in indexes}


def get_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=get_pinecone_api_key())
    if PINECONE_INDEX_HOST:
        return pc.Index(host=PINECONE_INDEX_HOST)

    if PINECONE_INDEX_NAME not in _list_pinecone_index_names(pc):
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSION, metric=PINECONE_METRIC, spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION), deletion_protection="disabled")
        deadline = time.time() + 120
        while time.time() < deadline:
            description = pc.describe_index(PINECONE_INDEX_NAME)
            status = getattr(description, "status", {}) or {}
            if status.get("ready"):
                break
            time.sleep(3)
    return pc.Index(PINECONE_INDEX_NAME)


def upsert_documents_to_pinecone(openai_client, index, documents):
    global _PINECONE_INDEXED
    if _PINECONE_INDEXED:
        return
    embeddings = embed_texts(openai_client, [f"{document['title']}\n{document['content']}" for document in documents])
    vectors = []
    for document, embedding in zip(documents, embeddings):
        vectors.append({"id": document["id"], "values": embedding, "metadata": {"title": document["title"], "content": document["content"]}})
    index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
    _PINECONE_INDEXED = True


def semantic_search(question, documents, top_k=TOP_K):
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_api_key())
    index = get_pinecone_index()
    if PINECONE_UPSERT_ON_QUERY:
        upsert_documents_to_pinecone(client, index, documents)
    question_embedding = embed_texts(client, [question])[0]
    result = index.query(vector=question_embedding, top_k=top_k, include_metadata=True, namespace=PINECONE_NAMESPACE)
    return pinecone_matches_to_documents(result)


def _pinecone_result_get(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def pinecone_matches_to_documents(result, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    context_documents = []
    for match in _pinecone_result_get(result, "matches", []):
        match_id = _pinecone_result_get(match, "id")
        if match_id in exclude_ids:
            continue
        metadata = _pinecone_result_get(match, "metadata", {}) or {}
        content = metadata.get("content") or metadata.get("text") or metadata.get("summary") or json.dumps(metadata)
        context_documents.append({"id": match_id, "title": metadata.get("title", match_id), "content": content, "score": _pinecone_result_get(match, "score"), "metadata": metadata})
    return context_documents


def _records_from_fetch(fetch_result):
    vectors = _pinecone_result_get(fetch_result, "vectors", {}) or {}
    if isinstance(vectors, dict):
        return vectors
    return {record.id: record for record in vectors}


def _record_to_seed_text(record):
    metadata = _pinecone_result_get(record, "metadata", {}) or {}
    return "\n".join(str(value) for value in [metadata.get("title"), metadata.get("content"), metadata.get("text"), metadata.get("summary")] if value)


def recommend_from_pinecone(payload, documents, top_k=TOP_K):
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_api_key())
    index = get_pinecone_index()
    if PINECONE_UPSERT_ON_QUERY:
        upsert_documents_to_pinecone(client, index, documents)

    seed_id = payload.get("seed_id")
    exclude_ids = set(payload.get("exclude_ids") or [])
    vector = None
    seed_text = payload.get("seed_text") or payload.get("question")

    if seed_id:
        exclude_ids.add(seed_id)
        records = _records_from_fetch(index.fetch(ids=[seed_id], namespace=PINECONE_NAMESPACE))
        seed_record = records.get(seed_id)
        if not seed_record:
            raise ValueError(f"Seed record {seed_id!r} was not found in Pinecone namespace {PINECONE_NAMESPACE!r}.")
        vector = _pinecone_result_get(seed_record, "values")
        seed_text = seed_text or _record_to_seed_text(seed_record)

    if vector is None:
        user_profile = payload.get("user_profile") or {}
        interests = payload.get("interests") or []
        if seed_text and (user_profile or interests):
            seed_text = json.dumps({"seed_text": seed_text, "user_profile": user_profile, "interests": interests}, sort_keys=True)
        elif not seed_text:
            seed_text = json.dumps({"user_profile": user_profile, "interests": interests}, sort_keys=True)
        if not seed_text or seed_text == '{"interests": [], "user_profile": {}}':
            raise ValueError("Recommendation requests must include seed_id, seed_text, question, user_profile, or interests.")
        vector = embed_texts(client, [seed_text])[0]

    result = index.query(vector=vector, top_k=top_k + len(exclude_ids), include_metadata=True, namespace=PINECONE_NAMESPACE)
    recommendations = pinecone_matches_to_documents(result, exclude_ids=exclude_ids)
    return recommendations[:top_k], seed_text


def retrieve_context(question, documents, top_k=TOP_K):
    if ENABLE_PINECONE and (PINECONE_API_KEY_SECRET_ARN or os.environ.get("PINECONE_API_KEY")):
        return semantic_search(question, documents, top_k=top_k), "pinecone"
    return retrieve(question, documents, top_k=top_k), "keyword"


def call_agent(client, agent, question, context_documents):
    response = client.responses.create(model=OPENAI_MODEL, instructions=agent["instructions"], input=json.dumps({"question": question, "retrieved_context": context_documents}), text={"format": {"type": "json_schema", "name": f"{agent['name']}_result", "schema": AGENT_OUTPUT_SCHEMA, "strict": True}}, max_output_tokens=MAX_OUTPUT_TOKENS)
    return json.loads(response.output_text)


def synthesize_answer(client, question, context_documents, agent_outputs):
    instructions = "You are the final coordinator for an AWS-hosted multi-agent RAG endpoint. Use only the retrieved document context and the three agent outputs. Answer the user's question clearly, cite section titles, and include safety notes. If the manual does not provide enough information, say so."
    response = client.responses.create(model=OPENAI_MODEL, instructions=instructions, input=json.dumps({"question": question, "retrieved_context": context_documents, "agent_outputs": agent_outputs}), text={"format": {"type": "json_schema", "name": "multi_agent_rag_answer", "schema": FINAL_OUTPUT_SCHEMA, "strict": True}}, max_output_tokens=MAX_OUTPUT_TOKENS)
    return json.loads(response.output_text)


def answer_question(question, context_documents, requested_agents=None):
    from openai import OpenAI
    client = OpenAI(api_key=get_openai_api_key())
    requested_agents = requested_agents or [agent["name"] for agent in AGENTS]
    selected_agents = [agent for agent in AGENTS if agent["name"] in requested_agents]
    if not selected_agents:
        raise ValueError("No valid agents were requested.")
    agent_outputs = []
    for agent in selected_agents:
        result = call_agent(client, agent, question, context_documents)
        agent_outputs.append({"agent": agent["name"], "role": agent["role"], "result": result})
    final = synthesize_answer(client, question, context_documents, agent_outputs)
    return agent_outputs, final


def response(status_code, body):
    return {"statusCode": status_code, "headers": {"content-type": "application/json", "access-control-allow-origin": "*"}, "body": json.dumps(body)}


def handler(event, context):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            if event.get("isBase64Encoded"):
                return response(400, {"error": "Base64 encoded bodies are not supported."})
            payload = json.loads(body or "{}")
        else:
            payload = body or {}
        mode = payload.get("mode", "answer")
        question = payload.get("question")
        if mode != "recommendations" and not question:
            return response(400, {"error": "Request JSON must include a non-empty question."})
        top_k = int(payload.get("top_k", TOP_K))
        requested_agents = payload.get("agents")
        if mode == "recommendations":
            documents = load_documents() if PINECONE_UPSERT_ON_QUERY else []
            recommendations, seed_text = recommend_from_pinecone(payload, documents, top_k=top_k)
            return response(200, {"mode": mode, "seed_id": payload.get("seed_id"), "seed_text": seed_text, "retrieval_source": "pinecone", "recommendations": recommendations})
        documents = load_documents()
        retrieved_context, retrieval_source = retrieve_context(question, documents, top_k=top_k)
        if mode == "semantic_search":
            return response(200, {"question": question, "document": str(KNOWLEDGE_PATH), "retrieval_source": retrieval_source, "retrieved_context": retrieved_context})
        agent_outputs, final = answer_question(question, retrieved_context, requested_agents=requested_agents)
        return response(200, {"question": question, "document": str(KNOWLEDGE_PATH), "retrieval_source": retrieval_source, "agents": agent_outputs, "answer": final["answer"], "steps": final["steps"], "safety_notes": final["safety_notes"], "citations": final["citations"], "agent_consensus": final["agent_consensus"], "retrieved_context": retrieved_context})
    except Exception as error:
        return response(500, {"error": str(error)})
