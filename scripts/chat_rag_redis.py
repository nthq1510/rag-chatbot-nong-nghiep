#!/usr/bin/env python3
"""Chatbot RAG có bộ nhớ hội thoại ngắn hạn bằng Redis.

Luồng xử lý một câu hỏi:
1. Đọc lịch sử hội thoại từ Redis theo session_id.
2. Encode câu hỏi hiện tại để truy xuất chunks liên quan từ Qdrant.
3. Ghép HISTORY + CONTEXT + QUESTION thành prompt.
4. Gọi generator Hugging Face hoặc Ollama để sinh câu trả lời.
5. Lưu lại user question và assistant answer vào Redis.

Redis chỉ lưu lịch sử trò chuyện; tri thức sản phẩm vẫn nằm ở Qdrant/context.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid

from benchmark_rag import (
    DEFAULT_QDRANT_PATH,
    LLM_MODELS,
    QueryEncoder,
    LLMGenerator,
    build_context,
    load_yaml,
    query_qdrant,
    resolve_path,
)


# Lớp gọi Ollama local để sinh câu trả lời thay vì dùng Hugging Face model.
class OllamaGenerator:
    """Generator gọi Ollama local qua HTTP API."""

    def __init__(self, model_name, base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt, max_new_tokens):
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý tư vấn thuốc và phân bón nông nghiệp. "
                        "Chỉ trả lời dựa trên CONTEXT được cung cấp. "
                        "Không bịa tên thuốc, hoạt chất, liều lượng hoặc thông tin không có trong context. "
                        "Nếu context không đủ dữ liệu, hãy nói rõ là chưa đủ dữ liệu."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": 0,
                "num_predict": max_new_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Open Ollama first, then run: ollama pull <model-name>"
            ) from exc

        message = result.get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Ollama returned an empty response: {result}")
        return content.strip()


# Lớp đọc/ghi/xóa lịch sử hội thoại trong Redis theo session_id.
class RedisConversationMemory:
    """Lớp quản lý lưu/đọc/xóa lịch sử hội thoại trong Redis."""

    def __init__(self, redis_url, namespace="rag_chat", ttl_seconds=86400, max_turns=6):
        try:
            import redis
        except ImportError as exc:
            raise SystemExit("Missing dependency: redis. Run: python3 -m pip install redis") from exc

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.namespace = namespace
        self.ttl_seconds = int(ttl_seconds)
        self.max_turns = int(max_turns)
        try:
            self.client.ping()
        except redis.RedisError as exc:
            raise SystemExit(
                f"Cannot connect to Redis at {redis_url}. "
                "Start Redis first, or pass --redis-url for your Redis service."
            ) from exc

    def key_for(self, session_id):
        return f"{self.namespace}:{session_id}:messages"

    def load(self, session_id):
        key = self.key_for(session_id)
        messages = [json.loads(item) for item in self.client.lrange(key, 0, -1)]
        if self.max_turns <= 0:
            return []
        return messages[-self.max_turns * 2 :]

    def append_turn(self, session_id, question, answer):
        key = self.key_for(session_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps({"role": "user", "content": question}, ensure_ascii=False))
        pipe.rpush(key, json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False))
        if self.max_turns > 0:
            pipe.ltrim(key, -self.max_turns * 2, -1)
        if self.ttl_seconds > 0:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def clear(self, session_id):
        self.client.delete(self.key_for(session_id))


# Chuyển danh sách message Redis thành đoạn lịch sử hội thoại đưa vào prompt.
def format_history(messages):
    """Định dạng lịch sử hội thoại thành text đưa vào prompt."""
    if not messages:
        return "Chưa có lịch sử hội thoại."

    labels = {
        "user": "Người dùng",
        "assistant": "Trợ lý",
    }
    return "\n".join(
        f"{labels.get(item.get('role'), item.get('role', 'unknown'))}: {item.get('content', '')}"
        for item in messages
    )


FOLLOW_UP_MARKERS = (
    "đó",
    "do",
    "này",
    "nay",
    "ấy",
    "ay",
    "trên",
    "tren",
    "vừa",
    "vua",
    "sản phẩm đó",
    "san pham do",
    "thuốc đó",
    "thuoc do",
    "cây đó",
    "cay do",
    "loại đó",
    "loai do",
    "liều đó",
    "lieu do",
    "như thế nào",
    "nhu the nao",
    "dùng như thế nào",
    "dung nhu the nao",
    "dùng bao nhiêu",
    "dung bao nhieu",
    "cách dùng",
    "cach dung",
)


# Kiểm tra câu hỏi hiện tại có khả năng đang dùng bộ nhớ hội thoại hay không.
def inspect_memory_usage(question, history_messages, answer=None):
    normalized_question = question.lower()
    matched_markers = [
        marker for marker in FOLLOW_UP_MARKERS if marker in normalized_question
    ]
    has_history = bool(history_messages)
    user_turns = [item for item in history_messages if item.get("role") == "user"]
    assistant_turns = [
        item for item in history_messages if item.get("role") == "assistant"
    ]

    if has_history and matched_markers:
        decision = "likely_used_memory"
        explanation = (
            "Câu hỏi có dấu hiệu hỏi nối tiếp và Redis có lịch sử hội thoại."
        )
    elif has_history:
        decision = "memory_available_but_question_independent"
        explanation = (
            "Redis có lịch sử, nhưng câu hỏi hiện tại không có dấu hiệu phụ thuộc ngữ cảnh rõ ràng."
        )
    else:
        decision = "no_memory_available"
        explanation = "Redis chưa có lịch sử cho session này trước khi trả lời."

    answer_overlap_with_history = False
    if answer and assistant_turns:
        normalized_answer = answer.lower()
        answer_overlap_with_history = any(
            item.get("content", "").strip().lower()[:80] in normalized_answer
            for item in assistant_turns
            if len(item.get("content", "").strip()) >= 30
        )

    return {
        "decision": decision,
        "explanation": explanation,
        "has_history_before_answer": has_history,
        "history_message_count": len(history_messages),
        "history_user_turn_count": len(user_turns),
        "history_assistant_turn_count": len(assistant_turns),
        "matched_follow_up_markers": matched_markers,
        "answer_overlap_with_previous_assistant": answer_overlap_with_history,
    }


# Chuyển một kết quả Qdrant point thành dict gọn để hiển thị/lưu JSON.
def format_point(rank, point):
    payload = point.payload or {}
    return {
        "rank": rank,
        "score": float(point.score),
        "chunk_id": payload.get("chunk_id"),
        "product_id": payload.get("product_id"),
        "ten_san_pham": payload.get("ten_san_pham"),
        "text": payload.get("text"),
    }


# Encode câu hỏi rồi lấy top-k chunks liên quan từ Qdrant.
def retrieve_chunks(client, encoder, collection_name, question, top_k):
    started = time.perf_counter()
    vector = encoder.encode(question)
    points = query_qdrant(client, collection_name, vector, top_k)
    chunks = [
        format_point(rank=index, point=point)
        for index, point in enumerate(points, start=1)
    ]
    return chunks, time.perf_counter() - started


# Ghép lịch sử Redis, context Qdrant và câu hỏi hiện tại thành prompt chat.
def build_chat_prompt(question, history_messages, retrieved_chunks):
    history = format_history(history_messages)
    context = build_context(retrieved_chunks)
    return f"""LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{history}

CONTEXT TRUY XUẤT TỪ CƠ SỞ TRI THỨC:
{context}

CÂU HỎI HIỆN TẠI:
{question}

Yêu cầu:
- Dùng lịch sử hội thoại để hiểu các đại từ hoặc câu hỏi nối tiếp như "cây đó", "sản phẩm trên", "dùng bao nhiêu".
- Chỉ dùng CONTEXT để khuyến nghị sản phẩm, hoạt chất, liều lượng, cây trồng và lưu ý an toàn.
- Nếu người dùng hỏi tiếp về thông tin đã nói trước đó nhưng CONTEXT hiện tại không đủ, hãy nói rõ phần nào chưa đủ dữ liệu.
- Trả lời bằng tiếng Việt, tự nhiên, ngắn gọn nhưng đủ ý.
"""


# Lớp điều phối toàn bộ chatbot: Redis memory + Qdrant retrieval + generator.
class ChatRAGRedis:
    """Điều phối RAG + Redis memory cho một phiên chat."""

    def __init__(
        self,
        config_path,
        qdrant_path,
        redis_url,
        session_id,
        embedding_alias,
        generator_backend,
        llm_alias,
        ollama_model,
        ollama_url,
        device,
        llm_device,
        top_k,
        max_new_tokens,
        memory_ttl_seconds,
        memory_turns,
    ):
        from qdrant_client import QdrantClient

        config = load_yaml(config_path)
        embedding_configs = [
            item for item in config["models"] if item["alias"] == embedding_alias
        ]
        if not embedding_configs:
            raise ValueError(f"Embedding alias not found in config: {embedding_alias}")

        if generator_backend == "hf" and llm_alias not in LLM_MODELS:
            raise ValueError(f"LLM alias not found: {llm_alias}")

        self.session_id = session_id
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.embedding_config = embedding_configs[0]
        self.memory = RedisConversationMemory(
            redis_url=redis_url,
            ttl_seconds=memory_ttl_seconds,
            max_turns=memory_turns,
        )
        self.client = QdrantClient(path=str(resolve_path(qdrant_path)))
        self.encoder = QueryEncoder(
            alias=self.embedding_config["alias"],
            model_name=self.embedding_config["model_name"],
            device=device,
            max_length=self.embedding_config.get("max_length"),
        )
        if generator_backend == "ollama":
            self.generator = OllamaGenerator(model_name=ollama_model, base_url=ollama_url)
        else:
            self.generator = LLMGenerator(model_name=LLM_MODELS[llm_alias], device=llm_device)

    def ask(self, question):
        history = self.memory.load(self.session_id)
        memory_debug_before = inspect_memory_usage(question, history)
        chunks, retrieval_seconds = retrieve_chunks(
            client=self.client,
            encoder=self.encoder,
            collection_name=self.embedding_config["collection_name"],
            question=question,
            top_k=self.top_k,
        )
        prompt = build_chat_prompt(question, history, chunks)
        started = time.perf_counter()
        answer = self.generator.generate(prompt, max_new_tokens=self.max_new_tokens)
        generation_seconds = time.perf_counter() - started
        memory_debug = inspect_memory_usage(question, history, answer=answer)
        self.memory.append_turn(self.session_id, question, answer)
        return {
            "session_id": self.session_id,
            "question": question,
            "answer": answer,
            "memory_debug_before_generation": memory_debug_before,
            "memory_debug": memory_debug,
            "retrieved_chunks": chunks,
            "latency": {
                "retrieval_seconds": round(retrieval_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
            },
        }

    def clear_memory(self):
        self.memory.clear(self.session_id)


# Hàm chạy từ terminal: đọc tham số CLI và thực hiện một lượt chat.
def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat with Redis conversation memory.")
    parser.add_argument("--config", default="configs/benchmark_models.yaml")
    parser.add_argument("--qdrant-path", default=str(DEFAULT_QDRANT_PATH))
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--embedding", default="bge-m3")
    parser.add_argument("--generator", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--llm", choices=sorted(LLM_MODELS), default="qwen3_4b")
    parser.add_argument("--ollama-model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--memory-ttl-seconds", type=int, default=86400)
    parser.add_argument("--memory-turns", type=int, default=6)
    parser.add_argument("--question", default=None, help="Ask one question and exit.")
    parser.add_argument("--clear-memory", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--show-memory-debug", action="store_true")
    args = parser.parse_args()

    session_id = args.session_id or uuid.uuid4().hex[:12]
    chat = ChatRAGRedis(
        config_path=args.config,
        qdrant_path=args.qdrant_path,
        redis_url=args.redis_url,
        session_id=session_id,
        embedding_alias=args.embedding,
        generator_backend=args.generator,
        llm_alias=args.llm,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        device=args.device,
        llm_device=args.llm_device,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        memory_ttl_seconds=args.memory_ttl_seconds,
        memory_turns=args.memory_turns,
    )

    if args.clear_memory:
        chat.clear_memory()
        print(f"Cleared memory for session_id={session_id}")

    if args.question:
        result = chat.ask(args.question)
        print(result["answer"])
        if args.show_memory_debug:
            print("\nKiểm tra memory:")
            print(json.dumps(result["memory_debug"], ensure_ascii=False, indent=2))
        if args.show_sources:
            print("\nNguồn truy xuất:")
            for chunk in result["retrieved_chunks"]:
                print(
                    f"- rank={chunk['rank']} score={chunk['score']:.4f} "
                    f"product_id={chunk.get('product_id')} ten_san_pham={chunk.get('ten_san_pham')}"
                )
        return

    print(f"session_id: {session_id}")
    print("Nhập câu hỏi. Gõ /clear để xoá nhớ, /exit để thoát.")
    while True:
        question = input("\nBạn: ").strip()
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            chat.clear_memory()
            print("Đã xoá lịch sử hội thoại.")
            continue
        result = chat.ask(question)
        print(f"\nTrợ lý: {result['answer']}")
        if args.show_memory_debug:
            print("\nKiểm tra memory:")
            print(json.dumps(result["memory_debug"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
