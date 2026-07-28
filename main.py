import base64
import json
import logging
import os
import re
import socket
import time
import uuid
import warnings
from io import BytesIO

import coloredlogs
import requests
import tiktoken
from flask import Flask, Response, jsonify, make_response, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymemcache.client.base import Client
from waitress import serve

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

try:
    import printedcolors
except Exception:
    class _Color:
        class fg:
            lightcyan = ""
        reset = ""
    printedcolors = type("printedcolors", (), {"Color": _Color})()

warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

logger = logging.getLogger("relay")
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

APP_NAME = os.getenv("APP_NAME", "relay")
PORT = int(os.getenv("PORT", os.getenv("APP_PORT", 5001)))
HOST = os.getenv("HOST", os.getenv("APP_HOST", "0.0.0.0"))

API_BASE = os.getenv("ONE_MIN_API_BASE", "https://api.1min.ai/api")
CHAT_URL = os.getenv("ONE_MIN_CHAT_API_URL", f"{API_BASE}/chat-with-ai")
FEATURES_URL = os.getenv("ONE_MIN_API_URL", f"{API_BASE}/features")
CONV_URL = os.getenv("ONE_MIN_CONVERSATION_API_URL", f"{API_BASE}/conversations")
STREAM_URL = os.getenv(
    "ONE_MIN_CONVERSATION_API_STREAMING_URL",
    f"{API_BASE}/chat-with-ai?isStreaming=true",
)
ASSET_URL = os.getenv("ONE_MIN_ASSET_URL", f"{API_BASE}/assets")

OLD_MODEL_LIST = [
    "gpt-5-nano",
    "gpt-5",
    "gpt-5-mini",
    "o3-mini",
    "deepseek-chat",
    "deepseek-reasoner",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-instant-1.2",
    "claude-2.1",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "open-mistral-7b",
    "gpt-o1-pro",
    "gpt-o4-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    "command",
]

VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"}

IMAGE_MODELS = {
    "stable-image",
    "stable-diffusion-xl-1024-v1-0",
    "stable-diffusion-v1-6",
    "esrgan-v1-x2plus",
    "clipdrop",
    "midjourney",
    "midjourney_6_1",
    "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
    "b24e16ff-06e3-43eb-8d33-4416c2d75876",
    "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
    "1e60896f-3c26-4296-8ecc-53e2afecc132",
    "aa77f04e-3eec-4034-9c07-d0f619684628",
    "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
    "black-forest-labs/flux-schnell",
}

MODEL_SUBSET = [
    m.strip()
    for m in os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS", "mistral-nemo,gpt-4o,deepseek-chat").split(",")
    if m.strip()
]
STRICT_MODELS = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY", "false").lower() == "true"
MODELS = MODEL_SUBSET if STRICT_MODELS else OLD_MODEL_LIST

MEM_HOST = os.getenv("MEMCACHED_HOST", os.getenv("MEM_HOST", "memcached"))
MEM_PORT = int(os.getenv("MEMCACHED_PORT", os.getenv("MEM_PORT", 11211)))

app = Flask(__name__)


def memcached_ok(host: str = MEM_HOST, port: int = MEM_PORT) -> bool:
    try:
        c = Client((host, port))
        c.set("relay_test", b"1")
        ok = c.get("relay_test") == b"1"
        c.delete("relay_test")
        return ok
    except Exception:
        return False


if memcached_ok():
    limiter = Limiter(get_remote_address, app=app, storage_uri=f"memcached://{MEM_HOST}:{MEM_PORT}")
else:
    limiter = Limiter(get_remote_address, app=app)
    logger.warning("Memcached unavailable; using in-memory rate limiting.")


def token_count(text: str, model: str = "gpt-4") -> int:
    try:
        if model.startswith("mistral"):
            tok = MistralTokenizer.from_model("open-mistral-nemo")
            req = ChatCompletionRequest(messages=[UserMessage(content=text)], model="open-mistral-nemo")
            return len(tok.encode_chat_completion(req).tokens)
        enc = tiktoken.encoding_for_model(model if model in {"gpt-3.5-turbo", "gpt-4"} else "gpt-4")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def error(code: int, model: str | None = None, key: str | None = None):
    mapping = {
        1002: ("The model does not exist.", "invalid_request_error", "model_not_found", 400),
        1020: (f"Incorrect API key provided: {key}.", "authentication_error", "invalid_api_key", 401),
        1021: ("Invalid Authentication", "invalid_request_error", None, 401),
        1212: ("Incorrect Endpoint. Please use the /v1/chat/completions endpoint.", "invalid_request_error", "model_not_supported", 400),
        1044: ("This model does not support image inputs.", "invalid_request_error", "model_not_supported", 400),
        1412: ("No message provided.", "invalid_request_error", "invalid_request_error", 400),
        1423: ("No content in last message.", "invalid_request_error", "invalid_request_error", 400),
        1405: ("Method Not Allowed", "invalid_request_error", None, 405),
    }
    msg, typ, err_code, http = mapping.get(code, ("Unknown error", "unknown_error", None, 400))
    if model:
        msg = msg.replace("does not exist.", f"does not exist: {model}.")
    payload = {"error": {"message": msg, "type": typ, "param": None, "code": err_code}}
    return jsonify(payload), http


def cors():
    resp = make_response()
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp, 204


def get_api_key():
    h = request.headers.get("Authorization", "")
    if not h.startswith("Bearer "):
        return None
    return h.split(" ", 1)[1].strip() or None


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def normalize_tools(data: dict):
    """Accept both the current `tools`/`tool_choice` shape and the legacy
    `functions`/`function_call` shape, and return them normalized as
    (tools, tool_choice) using the current shape."""
    tools = data.get("tools")
    tool_choice = data.get("tool_choice")
    if not tools and data.get("functions"):
        tools = [{"type": "function", "function": f} for f in data["functions"]]
        fc = data.get("function_call")
        if isinstance(fc, dict) and fc.get("name"):
            tool_choice = {"type": "function", "function": {"name": fc["name"]}}
        elif fc in ("auto", "none"):
            tool_choice = fc
    return tools, tool_choice


def build_tools_instructions(tools, tool_choice=None) -> str:
    """Describe the requested tools to the model using an old-style,
    prompt-based calling convention, since the upstream 1min.ai API has no
    native tool-calling support."""
    if not tools:
        return ""

    lines = [
        "You have access to the following tools/functions. If, and only if, you need "
        "to call one to fulfil the user's request, respond with nothing else but one "
        "or more blocks in exactly this format:",
        "<tool_call>",
        '{"name": "<tool name>", "arguments": <a JSON object matching the tool parameters>}',
        "</tool_call>",
        "",
        "You may emit several <tool_call> blocks if several tools are needed. Do not "
        "add any commentary before, between, or after them when calling a tool. If no "
        "tool call is required, answer normally in plain text without emitting any "
        "<tool_call> block.",
        "",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        lines.append(f"  parameters (JSON Schema): {json.dumps(params, ensure_ascii=False)}")

    if isinstance(tool_choice, dict):
        forced = tool_choice.get("function", {}).get("name")
        if forced:
            lines.append(f'\nYou MUST call the tool named "{forced}" now, using the format above.')
    elif tool_choice == "required":
        lines.append("\nYou MUST call one of the tools above now, using the format above.")
    elif tool_choice == "none":
        lines.append("\nDo not call any tool for this message; answer in plain text only.")

    return "\n".join(lines)


def extract_tool_calls(text: str):
    """Parse the old-style <tool_call> blocks out of a model completion and
    return (clean_text, tool_calls) in the OpenAI tool_calls shape."""
    calls = []
    for m in TOOL_CALL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        name = obj.get("name")
        if not name:
            continue
        args = obj.get("arguments", {})
        args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    clean = TOOL_CALL_RE.sub("", text or "").strip()
    return clean, (calls or None)


def render_tool_call_block(tool_call: dict) -> str:
    fn = tool_call.get("function", {})
    return f'<tool_call>\n{{"name": "{fn.get("name", "")}", "arguments": {fn.get("arguments", "{}")}}}\n</tool_call>'


def normalize_text(content):
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    txt = item["text"]
                    parts.append("".join(txt) if isinstance(txt, list) else str(txt))
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def build_prompt(messages, new_input="", tools=None, tool_choice=None):
    lines = []
    instructions = build_tools_instructions(tools, tool_choice)
    if instructions:
        lines.append(instructions)
        lines.append("")

    lines.append("Conversation History:\n")
    for msg in messages:
        role = str(msg.get("role", "")).capitalize()

        if msg.get("role") == "tool":
            content = normalize_text(msg.get("content", ""))
            lines.append(f"Tool Result (call {msg.get('tool_call_id', '')}):\n{content}")
            continue

        content = normalize_text(msg.get("content", ""))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            blocks = "\n".join(render_tool_call_block(tc) for tc in tool_calls)
            content = f"{content}\n{blocks}".strip() if content else blocks

        lines.append(f"{role}: {content}")

    if messages:
        lines.append("Respond like normal. Do not add role labels.")
        lines.append("User Message:\n")
    lines.append(new_input)
    return "\n".join(lines)


def upstream_error(resp):
    if resp.status_code == 401:
        return error(1020)
    try:
        j = resp.json()
        detail = j.get("message") or j.get("error") or j.get("detail") or resp.text[:500]
    except Exception:
        detail = resp.text[:500] if getattr(resp, "text", None) else "No response body"
    payload = {
        "error": {
            "message": f"1min.ai API error ({resp.status_code}): {detail}",
            "type": "upstream_error",
            "param": None,
            "code": f"upstream_{resp.status_code}",
        }
    }
    return jsonify(payload), resp.status_code


def set_headers(resp):
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["X-Request-ID"] = str(uuid.uuid4())


def upload_image(url: str, headers: dict) -> str:
    if url.startswith("data:image/png;base64,"):
        data = base64.b64decode(url.split(",", 1)[1])
        fileobj = BytesIO(data)
    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fileobj = BytesIO(r.content)
    files = {"asset": (f"relay-{uuid.uuid4()}", fileobj, "image/png")}
    r = requests.post(ASSET_URL, files=files, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["fileContent"]["path"]


def transform_chat(resp_json, model: str, prompt_tokens: int, has_tools: bool = False):
    text = resp_json["aiRecord"]["aiRecordDetail"]["resultObject"][0]

    tool_calls = None
    if has_tools:
        text, tool_calls = extract_tool_calls(text)

    comp_tokens = token_count(text, model)
    message = {"role": "assistant", "content": (text or None) if tool_calls else text}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens,
        },
    }


def iter_upstream_content(resp):
    for raw in resp.iter_lines(decode_unicode=False):
        if raw == b"":
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", errors="replace")

        if not line.startswith("data:"):
            continue

        data = line.split(":", 1)[1].strip()
        if data == "[DONE]":
            break

        try:
            parsed = json.loads(data)
            content = parsed.get("content") or parsed.get("delta", {}).get("content") or ""
        except Exception:
            content = data

        if content:
            yield content


def stream_chat(resp, model: str, prompt_tokens: int, has_tools: bool = False):
    stream_id = f"chatcmpl-{uuid.uuid4()}"
    chunks = []

    if not has_tools:
        for content in iter_upstream_content(resp):
            chunks.append(content)
            yield "data: " + json.dumps(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                },
                ensure_ascii=False,
            ) + "\n\n"

        full_text = "".join(chunks)
        comp_tokens = token_count(full_text, model)
        yield "data: " + json.dumps(
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": comp_tokens,
                    "total_tokens": prompt_tokens + comp_tokens,
                },
            },
            ensure_ascii=False,
        ) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    # When tools are requested, the <tool_call> markers must be fully
    # buffered before we can tell whether the completion is a tool call, so
    # we cannot forward incremental deltas as they arrive from upstream.
    for content in iter_upstream_content(resp):
        chunks.append(content)

    full_text = "".join(chunks)
    clean_text, tool_calls = extract_tool_calls(full_text)
    comp_tokens = token_count(clean_text or full_text, model)

    if tool_calls:
        delta = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        delta = {"role": "assistant", "content": clean_text}
        finish_reason = "stop"

    yield "data: " + json.dumps(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        },
        ensure_ascii=False,
    ) + "\n\n"
    yield "data: " + json.dumps(
        {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "total_tokens": prompt_tokens + comp_tokens,
            },
        },
        ensure_ascii=False,
    ) + "\n\n"
    yield "data: [DONE]\n\n"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        ip = socket.gethostbyname(socket.gethostname())
        return f"Congratulations! Your API is working!\n\nEndpoint: {ip}:{PORT}/v1"
    return error(1405)


@app.route("/v1/models", methods=["GET"])
@limiter.limit("500 per minute")
def models():
    data = [{"id": m, "object": "model", "owned_by": "1minai", "created": 1727389042} for m in MODELS]
    return jsonify({"data": data, "object": "list"})


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def chat():
    if request.method == "OPTIONS":
        return cors()

    api_key = get_api_key()
    if not api_key:
        return error(1021)

    data = request.get_json(silent=True) or {}
    messages = data.get("messages") or []
    if not messages:
        return error(1412)

    last = messages[-1].get("content")
    if not last:
        return error(1423)

    model = data.get("model", "mistral-nemo")
    if STRICT_MODELS and model not in MODELS:
        return error(1002, model)

    tools, tool_choice = normalize_tools(data)
    has_tools = bool(tools) and tool_choice != "none"

    headers = {"API-KEY": api_key, "Content-Type": "application/json"}
    prompt = build_prompt(messages, data.get("new_input", ""), tools=tools, tool_choice=tool_choice)

    attachments = None
    content = messages[-1].get("content")
    if isinstance(content, list):
        if model not in VISION_MODELS:
            return error(1044, model)
        image_paths = []
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if "text" in item:
                txt = item["text"]
                text_parts.append("".join(txt) if isinstance(txt, list) else str(txt))
            if "image_url" in item:
                try:
                    image_paths.append(upload_image(item["image_url"]["url"], headers))
                except Exception as e:
                    logger.exception("image upload failed: %s", e)
        if image_paths:
            attachments = {"images": image_paths}
        if text_parts:
            prompt = build_prompt(
                messages[:-1] + [{"role": "user", "content": "\n".join(text_parts)}],
                data.get("new_input", ""),
                tools=tools,
                tool_choice=tool_choice,
            )

    prompt_tokens = token_count(prompt, model)
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": model,
        "promptObject": {"prompt": prompt},
    }
    if attachments:
        payload["promptObject"]["attachments"] = attachments

    if not data.get("stream", False):
        r = requests.post(CHAT_URL, json=payload, headers=headers, timeout=120)
        if r.status_code != 200:
            return upstream_error(r)
        out = transform_chat(r.json(), model, prompt_tokens, has_tools=has_tools)
        resp = make_response(jsonify(out), 200)
        set_headers(resp)
        return resp

    r = requests.post(STREAM_URL, data=json.dumps(payload), headers=headers, stream=True, timeout=120)
    if r.status_code != 200:
        return upstream_error(r)
    return Response(stream_chat(r, model, prompt_tokens, has_tools=has_tools), content_type="text/event-stream")


@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("100 per minute")
def images():
    if request.method == "OPTIONS":
        return cors()

    api_key = get_api_key()
    if not api_key:
        return error(1021)

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt:
        return error(1412)

    model = data.get("model", "black-forest-labs/flux-schnell")
    if model not in IMAGE_MODELS:
        return error(1044, model)

    payload = {
        "type": "IMAGE_GENERATOR",
        "model": model,
        "promptObject": {
            "prompt": prompt,
            "n": data.get("n", 1),
            "size": data.get("size", "1024x1024"),
        },
    }
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    try:
        r = requests.post(f"{FEATURES_URL}?isStreaming=false", json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        urls = r.json()["aiRecord"]["aiRecordDetail"]["resultObject"]
        return jsonify({"created": int(time.time()), "data": [{"url": u} for u in urls]})
    except Exception as e:
        logger.exception("image generation failed: %s", e)
        return error(1044, model)


if __name__ == "__main__":
    ip = socket.gethostbyname(socket.gethostname())
    pub = requests.get("https://api.ipify.org", timeout=10).text

    # Startup banner (simple “ad”)
    logger.info(
        f"""{printedcolors.Color.fg.lightcyan}
====================================================
Enjoying this self-hosted relay?
You can get a hosted, managed version at:
  https://shop.kokodev.cc
with extra features like video generation,
failover nodes, and more.
====================================================
{printedcolors.Color.reset}"""
    )

    logger.info(
        f"""{printedcolors.Color.fg.lightcyan}
Server ready:
Internal IP: {ip}:{PORT}
Public IP: {pub}
OpenAI-compatible endpoint:
{ip}:{PORT}/v1
{printedcolors.Color.reset}"""
    )

    serve(app, host=HOST, port=PORT, threads=int(os.getenv("THREADS", "6")))
