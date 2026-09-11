"""Client gọi model qua API tương thích OpenAI (/v1/chat/completions).

Endpoint và API key đọc từ .env, không hard-code trong source.
Tương đương lệnh curl:

    curl -X POST "$LLM_BASE_URL/v1/chat/completions" \
      -H "Authorization: Bearer $LLM_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"model": "...", "messages": [{"role":"user","content":"Hello!"}]}'
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

import requests

from . import config

T = TypeVar("T")
R = TypeVar("R")


class LLMError(RuntimeError):
    pass


def build_payload(
    model: str,
    messages: Sequence[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
) -> Dict[str, Any]:
    """Dựng body request. Tách riêng để test được mà không cần gọi mạng."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }


def extract_content(response_json: Dict[str, Any]) -> str:
    """Lấy text trả về; báo lỗi rõ ràng thay vì KeyError khó hiểu."""
    try:
        choices = response_json["choices"]
        if not choices:
            raise LLMError("Response không có 'choices': {}".format(response_json))
        message = choices[0]["message"]
    except (KeyError, TypeError) as exc:
        raise LLMError("Response sai định dạng: {}".format(response_json)) from exc

    content = message.get("content")
    if content is None:
        raise LLMError("Message không có 'content': {}".format(message))
    return content


def strip_code_fence(text: str) -> str:
    """Model hay bọc JSON trong ```json ... ``` — gỡ ra trước khi parse."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON, chịu được cả khi model kèm chữ thừa quanh object."""
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise LLMError("Không parse được JSON từ response:\n{}".format(text[:500]))


class LLMClient:
    """Client có retry + backoff, an toàn khi dùng đa luồng."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.base_url = (base_url or config.require("LLM_BASE_URL")).rstrip("/")
        # Một số endpoint nội bộ không kiểm tra key, nên cho phép để trống.
        self.api_key = api_key if api_key is not None else (config.get("LLM_API_KEY") or "")
        self.model = model or config.require("LLM_MODEL")
        self.timeout = timeout if timeout is not None else config.get_int("LLM_TIMEOUT", 120)
        self.max_retries = max_retries if max_retries is not None else config.get_int("LLM_MAX_RETRIES", 4)
        self._local = threading.local()

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return self.base_url + "/chat/completions"
        return self.base_url + "/v1/chat/completions"

    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        payload = build_payload(
            self.model, messages,
            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
        )
        headers = build_headers(self.api_key)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    self.endpoint, headers=headers, json=payload, timeout=self.timeout,
                )
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise LLMError("HTTP {}: {}".format(resp.status_code, resp.text[:300]))
                if resp.status_code >= 400:
                    # Lỗi 4xx khác là lỗi request của ta — retry vô ích.
                    raise LLMError("HTTP {}: {}".format(resp.status_code, resp.text[:300]))
                return extract_content(resp.json())
            except LLMError as exc:
                last_error = exc
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                sleep_s = min(2 ** attempt, 16) * (1.0 + 0.25 * random.random())
                time.sleep(sleep_s)

        raise LLMError("Thất bại sau {} lần thử: {}".format(self.max_retries, last_error))

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Any:
        text = self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True,
        )
        return parse_json_response(text)


def run_parallel(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    workers: int = 8,
    desc: str = "",
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> List[Optional[R]]:
    """Chạy fn trên từng item song song, giữ nguyên thứ tự đầu ra.

    Item lỗi trả về None và được đếm vào log thay vì làm sập cả mẻ.
    """
    results: List[Optional[R]] = [None] * len(items)
    done = 0
    failed = 0
    lock = threading.Lock()
    total = len(items)

    def worker(index: int) -> None:
        nonlocal done, failed
        try:
            value = fn(items[index])
            results[index] = value
        except Exception as exc:  # noqa: BLE001 - một item hỏng không được dừng cả mẻ
            with lock:
                failed += 1
            if on_error is not None:
                on_error(items[index], exc)
        finally:
            with lock:
                done += 1
                if desc and (done % 25 == 0 or done == total):
                    print("  {} {}/{} (lỗi: {})".format(desc, done, total, failed), flush=True)

    if total == 0:
        return results

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(worker, range(total)))

    return results
