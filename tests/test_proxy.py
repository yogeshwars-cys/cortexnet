import json
import os
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cortexnet.proxy import Proxy, Rewriter, classify, find_cwd

BIG = "\n".join(f"def f{i}(x):\n    y = x * {i}\n    z = y + {i}\n    return z - {i}\n\n" for i in range(120))


def numbered(text):
    return "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(text.split("\n"), 1))


def anthropic(path, extra_turn=False, call_id="t1", input_=None):
    msgs = [{"role": "user", "content": "look at the file"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": "Read",
                                               "input": input_ or {"file_path": path}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                          "content": [{"type": "text", "text": numbered(open(path).read())},
                                                      {"type": "text", "text": "<system-reminder>r</system-reminder>",
                                                       "cache_control": {"type": "ephemeral"}}]}]}]
    if extra_turn:
        msgs += [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                 {"role": "user", "content": "next"}]
    return {"model": "m", "system": "Primary working directory: /tmp", "messages": msgs}


def result(body):
    return body["messages"][2]["content"][0]["content"][0]["text"]


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.big = os.path.join(self.dir, "big.py")
        self.small = os.path.join(self.dir, "small.py")
        with open(self.big, "w") as f:
            f.write(BIG)
        with open(self.small, "w") as f:
            f.write("x = 1\n")
        self.rw = Rewriter(os.path.join(self.dir, "state"))

    def tearDown(self):
        self.tmp.cleanup()

    def send(self, body, rw=None):
        raw, stats = (rw or self.rw).rewrite(json.dumps(body).encode())
        return json.loads(raw), stats

    def test_outline_then_byte_identical_replay(self):
        body, stats = self.send(anthropic(self.big))
        text = result(body)
        self.assertTrue(text.startswith("[cortexnet]"))
        self.assertEqual(stats["rewrites"], 1)
        self.assertLess(stats["new_tokens"], stats["orig_tokens"] / 2)
        self.assertIn("<system-reminder>r</system-reminder>", text)
        self.assertEqual(body["messages"][2]["content"][0]["content"][0]["cache_control"], {"type": "ephemeral"})
        later, stats = self.send(anthropic(self.big, extra_turn=True))
        self.assertEqual(result(later), text)
        self.assertEqual(stats["rewrites"], 0)
        restarted, _ = self.send(anthropic(self.big, extra_turn=True), Rewriter(os.path.join(self.dir, "state")))
        self.assertEqual(result(restarted), text)

    def test_untouched_cases(self):
        original = numbered(BIG)
        old, _ = self.send(anthropic(self.big, extra_turn=True))                     # never seen fresh
        self.assertEqual(result(old), original)
        small, _ = self.send(anthropic(self.small, call_id="t2"))
        self.assertFalse(result(small).startswith("[cortexnet]"))
        ranged, _ = self.send(anthropic(self.big, call_id="t3", input_={"file_path": self.big, "offset": 5,
                                                                         "limit": 10}))
        self.assertEqual(result(ranged), original)
        first, _ = self.send(anthropic(self.big, call_id="t4"))
        self.assertTrue(result(first).startswith("[cortexnet]"))
        again, _ = self.send(anthropic(self.big, call_id="t5"))                      # second whole read
        self.assertEqual(result(again), original)

    def test_aware_announces_and_reads_intent(self):
        seen = {}
        rw = Rewriter(os.path.join(self.dir, "aw"), aware=True)
        rw.reader.read = lambda path, query="", snapshot="", **k: seen.update(q=query, s=snapshot) or "outline"
        body = anthropic(self.big)
        body["messages"][0]["content"] = "fix the clamp in f7"
        body["messages"][1]["content"] = [{"type": "thinking", "thinking": "f7 multiplies wrong"},
                                          {"type": "text", "text": "Reading big.py for f7."},
                                          {"type": "tool_use", "id": "g1", "name": "Grep", "input": {"pattern": "f7"}}
                                          ] + body["messages"][1]["content"]
        out, stats = self.send(body, rw)
        self.assertEqual(stats["rewrites"], 1)
        self.assertIn("fix the clamp in f7", seen["q"])
        self.assertIn('"pattern": "f7"', seen["q"])                       # sibling call in the same turn
        self.assertIn("f7 multiplies wrong", seen["s"])
        self.assertIn("Reading big.py for f7.", seen["s"])
        self.assertTrue(out["system"].endswith("second time returns the full file."))
        again, _ = self.send(anthropic(self.big, extra_turn=True), rw)
        self.assertEqual(again["system"], out["system"])                   # same bytes every request

    def test_passthrough_is_byte_identical(self):
        raw = json.dumps(anthropic(self.big)).encode()
        out, stats = Rewriter(os.path.join(self.dir, "pt"), rewrite=False).rewrite(raw)
        self.assertEqual(out, raw)
        self.assertEqual(stats["model"], "m")

    def test_openai_chat_and_responses(self):
        chat = {"model": "m", "messages": [
            {"role": "system", "content": f"<cwd>{self.dir}</cwd>"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "big.py"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": BIG}]}
        body, stats = self.send(chat)
        self.assertTrue(body["messages"][2]["content"].startswith("[cortexnet]"))
        resp = {"model": "m", "instructions": f"<cwd>{self.dir}</cwd>", "input": [
            {"type": "function_call", "call_id": "s1", "name": "shell",
             "arguments": json.dumps({"command": ["bash", "-lc", "cat ../" + os.path.basename(self.dir) + "/big.py"]})},
            {"type": "function_call_output", "call_id": "s1", "output": BIG}]}
        body, stats = self.send(resp, Rewriter(os.path.join(self.dir, "s2")))
        self.assertTrue(body["input"][1]["output"].startswith("[cortexnet]"))

    def test_gemini_cloud_code_with_mcp_wrapper(self):
        def call(cid, name, args):
            return {"role": "model", "parts": [{"functionCall": {"id": cid, "name": name, "args": args},
                                                "thoughtSignature": "sig"}]}

        def resp(cid, name, text):
            return {"role": "user", "parts": [{"functionResponse": {"id": cid, "name": name,
                                                                    "response": {"output": text}}}]}

        serena = {"ServerName": "serena", "ToolName": "read_file", "Arguments": json.dumps({"relative_path": "big.py"})}
        overview = {"ServerName": "serena", "ToolName": "get_symbols_overview", "Arguments": {"relative_path": "big.py"}}
        body = {"model": "gemini", "project": "p", "request": {
            "systemInstruction": {"parts": [{"text": f"Working directory: {self.dir}"}]},
            "contents": [{"role": "user", "parts": [{"text": "go"}]},
                         call("a", "call_mcp_tool", overview), resp("a", "call_mcp_tool", BIG),
                         call("b", "call_mcp_tool", serena), resp("b", "call_mcp_tool", BIG)]}}
        out, stats = self.send(body)
        contents = out["request"]["contents"]
        self.assertEqual(contents[2]["parts"][0]["functionResponse"]["response"]["output"], BIG)   # not fresh
        self.assertTrue(contents[4]["parts"][0]["functionResponse"]["response"]["output"].startswith("[cortexnet]"))
        self.assertEqual(contents[3]["parts"][0]["thoughtSignature"], "sig")
        body["request"]["contents"] = body["request"]["contents"][:3]
        out, _ = self.send(body, Rewriter(os.path.join(self.dir, "g2")))                   # symbol tool: untouched
        self.assertEqual(out["request"]["contents"][2]["parts"][0]["functionResponse"]["response"]["output"], BIG)
        self.assertEqual(classify("view_file", {"AbsolutePath": self.big}), ("whole", os.path.normpath(self.big)))
        self.assertEqual(classify("view_file", {"AbsolutePath": self.big, "StartLine": 3, "EndLine": 9})[2:], (3, 9))

    def test_classify_and_cwd(self):
        self.assertEqual(classify("Edit", {"file_path": "/a.py"}), None)
        self.assertEqual(classify("cortex_read", {"path": "/a.py"}), None)
        self.assertEqual(classify("Bash", {"command": "cat a.py | head"}), None)
        sh = lambda c: classify("shell_command", {"command": c}, "/w")   # Codex on Windows: PowerShell
        self.assertEqual(sh("Get-Content -Path a.py -Raw"), ("whole", os.path.normpath("/w/a.py")))
        self.assertEqual(sh("powershell.exe -NoProfile -Command \"Get-Content 'a.py'\"")[0], "whole")
        self.assertEqual(sh("Get-Content a.py -TotalCount 200")[2:], (1, 200))
        self.assertEqual(sh("Get-Content a.py | Select-Object -Skip 99 -First 50")[2:], (100, 149))
        self.assertEqual(sh("Get-Content a.py | Select-String foo"), None)
        self.assertEqual(sh("Get-Content a.py; rm a.py"), None)
        self.assertEqual(classify("Bash", {"command": "sed -n '3,9p' a.py"}, "/w")[1:],
                         (os.path.normpath("/w/a.py"), 3, 9))
        self.assertEqual(classify("str_replace_editor", {"command": "view", "path": "/a.py", "view_range": [2, 4]})[2:],
                         (2, 4))
        self.assertEqual(find_cwd({"system": [{"type": "text", "text": " - Primary working directory: /x/y\n"}]}),
                         os.path.normpath("/x/y") if os.name != "nt" else os.path.normpath("X:/y"))

    def test_drop_tools_is_static(self):
        rw = Rewriter(os.path.join(self.dir, "dt"), drop_tools=r"^mcp__blender__")
        body = {"model": "m", "tools": [{"name": "Read"}, {"name": "mcp__blender__look"}], "messages": []}
        out, stats = self.send(body, rw)
        self.assertEqual([t["name"] for t in out["tools"]], ["Read"])
        self.assertEqual(stats["dropped_tools"], 1)

    def test_http_roundtrip_streams_and_logs_usage(self):
        seen = {}
        sse = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":3,'
               b'"cache_creation_input_tokens":10,"cache_read_input_tokens":200,"output_tokens":1}}}\n\n'
               b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":42}}\n\n')

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                seen["path"] = self.path
                seen["auth"] = self.headers.get("x-api-key")
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(sse)))
                self.end_headers()
                self.wfile.write(sse)

        up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        px = Proxy(f"http://127.0.0.1:{up.server_address[1]}", self.rw, port=0).start()
        try:
            conn = HTTPConnection("127.0.0.1", px.port, timeout=10)
            conn.request("POST", "/v1/messages", body=json.dumps(anthropic(self.big)),
                         headers={"Content-Type": "application/json", "x-api-key": "k"})
            resp = conn.getresponse()
            self.assertEqual((resp.status, resp.read()), (200, sse))
        finally:
            px.stop()
            up.shutdown()
        self.assertEqual((seen["path"], seen["auth"]), ("/v1/messages", "k"))
        self.assertTrue(result(seen["body"]).startswith("[cortexnet]"))
        with open(self.rw.usage_path) as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["rewrites"], 1)
        self.assertEqual(rec["usage"], {"input_tokens": 3, "cache_creation_input_tokens": 10,
                                        "cache_read_input_tokens": 200, "output_tokens": 42})
        self.assertNotIn("big.py", json.dumps(rec["usage"]))

    def test_chunked_request_body(self):
        seen = {}

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=up.serve_forever, daemon=True).start()
        px = Proxy(f"http://127.0.0.1:{up.server_address[1]}", self.rw, port=0).start()
        try:
            conn = HTTPConnection("127.0.0.1", px.port, timeout=10)
            conn.request("POST", "/x", body=iter([b'{"a":', b' 1}']), encode_chunked=True,
                         headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"})
            self.assertEqual(conn.getresponse().read(), b"ok")
        finally:
            px.stop()
            up.shutdown()
        self.assertEqual(seen["body"], b'{"a": 1}')


if __name__ == "__main__":
    unittest.main()
