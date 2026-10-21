"""ServerLLM against a fake OpenAI-compatible server (no model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from run_assistant import ServerLLM, run_conversation

SEEN: list[dict] = []


class FakeServer(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(body)
        msgs = body["messages"]
        if not any(m["role"] == "tool" for m in msgs):
            args = {"steps": [{"id": "cac", "formula": "cac", "inputs": {"sm_spend": 500000, "new_customers": 250}}]}
            msg = {"role": "assistant", "content": "", "reasoning_content": "CAC = S&M / new customers.",
                   "tool_calls": [{"id": "x1", "type": "function",
                                   "function": {"name": "finance_calculator", "arguments": json.dumps(args)}}]}
        else:
            result = json.loads(msgs[-1]["content"])
            msg = {"role": "assistant", "reasoning_content": "Copy the value.",
                   "content": f"## CFO Report\n| CAC | {result['results']['cac']['display']} |"}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_server_backend_round_trip():
    srv = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = ServerLLM(f"http://127.0.0.1:{srv.server_port}/v1", "fake")
        trace = run_conversation(llm, "<financial_context>| S&M | 500,000 | New | 250 |</financial_context>\n\nQuestion: CAC?",
                                 max_new_tokens=64, temperature=0.0)
    finally:
        srv.shutdown()
    assert trace["thinks"] == ["CAC = S&M / new customers.", "Copy the value."]
    assert trace["calls"][0]["result"]["results"]["cac"]["display"] == "2,000"
    assert trace["calls"][0]["unsupported_inputs"] == []
    assert "| CAC | 2,000 |" in trace["report"]
    # second request carried the assistant tool call and the matching tool result in OpenAI format
    second = SEEN[-1]["messages"]
    asst = next(m for m in second if m["role"] == "assistant")
    tool = next(m for m in second if m["role"] == "tool")
    assert isinstance(asst["tool_calls"][0]["function"]["arguments"], str)
    assert tool["tool_call_id"] == asst["tool_calls"][0]["id"]
    assert SEEN[0]["tools"][0]["function"]["name"] == "finance_calculator"


def test_period_guard_answers_without_model():
    from run_assistant import period_guard

    known = {"Apr-25", "Mar-26", "Q4 FY26"}
    trace = period_guard("What was our LTV:CAC for Mar-25?", known, {"mis.xlsx"})
    assert trace is not None and trace["calls"] == [] and "Mar-25" in trace["report"]
    assert period_guard("What was CM2 for Mar-26?", known) is None
