#!/usr/bin/env python3
"""
Lightweight proxy between OpenClaw and mlx_vlm.
Logs all request fields, forwards to upstream, and if upstream
returns 422, strips unsupported fields and retries automatically.
"""

import argparse
import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

UPSTREAM = "http://127.0.0.1:10012"

# --- Feature toggles (defaults, overridden by CLI args) ---
ENABLE_STRIP_FIELDS = False
ENABLE_RENAME_FIELDS = True
ENABLE_NORMALIZE_MESSAGES = False
ENABLE_STRIP_MODEL = False

STRIP_FIELDS = {
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "logprobs",
    "top_logprobs",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "service_tier",
    "store",
    "metadata",
    "stream_options",
}

RENAME_FIELDS = {
    "max_completion_tokens": "max_tokens",
}


def summarize_value(v):
    """Short summary of a value for logging (avoid dumping huge content)."""
    if isinstance(v, str):
        return f"str(len={len(v)})"
    if isinstance(v, list):
        return f"list(len={len(v)})"
    if isinstance(v, dict):
        return f"dict(keys={list(v.keys())})"
    return repr(v)


def normalize_messages(messages):
    """
    Make messages compatible with mlx_vlm:
    - Convert role "tool" to "user" (with tool call context)
    - Convert array-style content blocks to plain strings
    - Fix null content
    - Remove tool_calls from assistant messages
    - Strip unknown message fields
    """
    ALLOWED_FIELDS = {"role", "content", "name"}
    normalized = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Convert content
        if content is None:
            content = ""
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            name = msg.get("name", "")
            prefix = f"[Tool result"
            if name:
                prefix += f" from {name}"
            if tool_call_id:
                prefix += f" (call_id: {tool_call_id})"
            prefix += "]"
            content = f"{prefix}\n{content}" if content else prefix
            role = "user"

        if role == "assistant" and not content:
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                calls_text = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    calls_text.append(
                        f'{fn.get("name", "?")}({fn.get("arguments", "")})'
                    )
                content = "[Tool calls: " + ", ".join(calls_text) + "]"

        clean = {"role": role, "content": content}
        if "name" in msg and role != "tool":
            clean["name"] = msg["name"]
        normalized.append(clean)

    # Collapse consecutive user messages (mlx_vlm may reject them)
    merged = []
    for msg in normalized:
        if merged and merged[-1]["role"] == msg["role"] == "user":
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)

    return merged


def forward(url, payload_bytes, timeout=600):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        log.info("=" * 60)
        log.info("=== Raw Request: %s %s", self.command, self.path)
        log.info("--- Headers ---")
        for header, value in self.headers.items():
            log.info("  %s: %s", header, value)
        log.info("--- Body (%d bytes) ---", len(body))
        try:
            raw_json = json.loads(body)
            log.info("%s", json.dumps(raw_json, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.info("%s", body.decode("utf-8", errors="replace"))
        log.info("=" * 60)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        field_summary = {k: summarize_value(v) for k, v in data.items()}
        log.info("=== Incoming: %s", json.dumps(field_summary))

        upstream_url = f"{UPSTREAM}{self.path}"

        removed = []
        renamed = []
        msg_count_before = msg_count_after = len(data.get("messages", []))

        if ENABLE_STRIP_MODEL and "model" in data:
            log.info("Stripping 'model' field (was: %s)", data["model"])
            data.pop("model")

        if ENABLE_STRIP_FIELDS:
            removed = [k for k in list(data.keys()) if k in STRIP_FIELDS]
            for key in STRIP_FIELDS:
                data.pop(key, None)

        if ENABLE_RENAME_FIELDS:
            for old_key, new_key in RENAME_FIELDS.items():
                if old_key in data:
                    data[new_key] = data.pop(old_key)
                    renamed.append(f"{old_key}->{new_key}")

        if ENABLE_NORMALIZE_MESSAGES and "messages" in data:
            msg_count_before = len(data["messages"])
            data["messages"] = normalize_messages(data["messages"])
            msg_count_after = len(data["messages"])

        if removed or renamed or msg_count_before != msg_count_after:
            log.info(
                "Cleaned: stripped=%s, renamed=%s, msgs %d->%d",
                removed,
                renamed,
                msg_count_before,
                msg_count_after,
            )

        cleaned = json.dumps(data).encode()

        try:
            resp = forward(upstream_url, cleaned)
            resp_body = resp.read()
            log.info("Upstream returned %d (size=%d)", resp.status, len(resp_body))
            self._send_response(resp.status, resp.getheaders(), resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            log.error("Upstream error %d: %s", e.code, err_body[:2000])
            self._send_response(e.code, [], err_body)
        except (BrokenPipeError, ConnectionResetError) as e:
            log.warning("Client disconnected before response: %s", e)
        except Exception as e:
            log.error("Upstream error: %s", e)
            try:
                self.send_error(502, str(e))
            except (BrokenPipeError, ConnectionResetError):
                log.warning("Client already disconnected, cannot send 502")

    def _send_response(self, status, headers, body):
        try:
            self.send_response(status)
            for key, val in headers or []:
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            if not any(k.lower() == "content-type" for k, _ in (headers or [])):
                self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as e:
            log.warning("Client disconnected during response write: %s", e)

    def do_GET(self):
        upstream_url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(upstream_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                self._send_response(resp.status, resp.getheaders(), resp.read())
        except urllib.error.HTTPError as e:
            self._send_response(e.code, [], e.read())
        except (BrokenPipeError, ConnectionResetError) as e:
            log.warning("Client disconnected (GET): %s", e)

    def log_message(self, format, *args):
        log.info("%s %s", self.address_string(), format % args)


class QuietHTTPServer(HTTPServer):
    def handle_error(self, request, client_address):
        exc_type = sys.exc_info()[0]
        if exc_type in (BrokenPipeError, ConnectionResetError):
            log.warning("Connection lost from %s (ignored)", client_address[0])
        else:
            super().handle_error(request, client_address)


def main():
    global UPSTREAM, ENABLE_STRIP_FIELDS, ENABLE_RENAME_FIELDS, ENABLE_NORMALIZE_MESSAGES, ENABLE_STRIP_MODEL
    parser = argparse.ArgumentParser(description="mlx_vlm compatibility proxy")
    parser.add_argument("--port", type=int, default=10010, help="Proxy listen port")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy bind address")
    parser.add_argument("--upstream", default=UPSTREAM, help="mlx_vlm server URL")

    strip_grp = parser.add_mutually_exclusive_group()
    strip_grp.add_argument(
        "--strip-fields",
        dest="strip_fields",
        action="store_true",
        default=None,
        help="Enable stripping unsupported fields",
    )
    strip_grp.add_argument(
        "--no-strip-fields",
        dest="strip_fields",
        action="store_false",
        help="Disable stripping unsupported fields",
    )

    rename_grp = parser.add_mutually_exclusive_group()
    rename_grp.add_argument(
        "--rename-fields",
        dest="rename_fields",
        action="store_true",
        default=None,
        help="Enable renaming fields (e.g. max_completion_tokens -> max_tokens)",
    )
    rename_grp.add_argument(
        "--no-rename-fields",
        dest="rename_fields",
        action="store_false",
        help="Disable renaming fields",
    )

    norm_grp = parser.add_mutually_exclusive_group()
    norm_grp.add_argument(
        "--normalize-messages",
        dest="normalize_messages",
        action="store_true",
        default=None,
        help="Enable message normalization for mlx_vlm compatibility",
    )
    norm_grp.add_argument(
        "--no-normalize-messages",
        dest="normalize_messages",
        action="store_false",
        help="Disable message normalization",
    )

    model_grp = parser.add_mutually_exclusive_group()
    model_grp.add_argument(
        "--strip-model",
        dest="strip_model",
        action="store_true",
        default=None,
        help="Enable stripping the 'model' field from request body",
    )
    model_grp.add_argument(
        "--no-strip-model",
        dest="strip_model",
        action="store_false",
        help="Disable stripping the 'model' field",
    )

    args = parser.parse_args()

    UPSTREAM = args.upstream
    if args.strip_fields is not None:
        ENABLE_STRIP_FIELDS = args.strip_fields
    if args.rename_fields is not None:
        ENABLE_RENAME_FIELDS = args.rename_fields
    if args.normalize_messages is not None:
        ENABLE_NORMALIZE_MESSAGES = args.normalize_messages
    if args.strip_model is not None:
        ENABLE_STRIP_MODEL = args.strip_model

    log.info(
        "Config: strip_fields=%s, rename_fields=%s, normalize_messages=%s, strip_model=%s",
        ENABLE_STRIP_FIELDS,
        ENABLE_RENAME_FIELDS,
        ENABLE_NORMALIZE_MESSAGES,
        ENABLE_STRIP_MODEL,
    )

    server = QuietHTTPServer((args.host, args.port), ProxyHandler)
    log.info("Proxy listening on %s:%d -> %s", args.host, args.port, UPSTREAM)
    server.serve_forever()


if __name__ == "__main__":
    main()
