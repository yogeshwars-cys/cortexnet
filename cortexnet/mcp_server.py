"""Minimal MCP server (stdio, newline-delimited JSON-RPC 2.0) exposing the outline-first reader.

No dependencies beyond the standard library and numpy. Register with an MCP client, e.g. Claude Code:
    claude mcp add cortexnet -- python -m cortexnet.mcp_server
State (include model, event log) goes to $CORTEXNET_STATE or ./.cortexnet. $CORTEXNET_SPLIT (default 600, 0 = off):
leaf symbols bigger than this many tokens get sub-blocks in the outline, so a miss can fetch one block.
"""
import json
import os
import sys
import traceback

from .reader import Reader

APPROVE = {"type": "array", "items": {"type": "string"},
           "description": "menu ids from a [connected] list to pull in the same call, e.g. [\"#2\"]"}
TOOLS = [
    {"name": "cortex_read",
     "description": "Read a file outline-first. Small files come back whole. Large files come back as an outline "
                    "(symbols with line spans) plus the bodies most likely needed. Use cortex_range for any other lines.",
     "inputSchema": {"type": "object", "required": ["path"], "properties": {
         "path": {"type": "string", "description": "file path"},
         "query": {"type": "string", "description": "what you are looking for: symbol names, error text, intent"},
         "approve": APPROVE}}},
    {"name": "cortex_range",
     "description": "Return numbered lines start..end (1-based, inclusive) of a file.",
     "inputSchema": {"type": "object", "required": ["path", "start", "end"], "properties": {
         "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "approve": APPROVE}}},
    {"name": "cortex_approve",
     "description": "Pull the bodies of connected symbols offered in a [connected] menu, by id (e.g. [\"#3\"]).",
     "inputSchema": {"type": "object", "required": ["ids"], "properties": {"ids": APPROVE}}},
    {"name": "cortex_outline",
     "description": "Return only the outline of a file: symbols, line spans, signatures.",
     "inputSchema": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}}},
    {"name": "cortex_stats",
     "description": "Token use so far: tokens shown vs whole-file reads, range misses, include-model updates.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def handle(reader, req):
    m, params = req.get("method"), req.get("params") or {}
    if m == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"), "capabilities": {"tools": {}},
                "serverInfo": {"name": "cortexnet", "version": "0.2.0"}}
    if m == "ping":
        return {}
    if m == "tools/list":
        return {"tools": TOOLS}
    if m == "tools/call":
        name, a = params.get("name"), params.get("arguments") or {}
        try:
            if name == "cortex_read":
                text = reader.read(a["path"], a.get("query", ""), a.get("approve") or ())
            elif name == "cortex_range":
                text = reader.range(a["path"], a["start"], a["end"], a.get("approve") or ())
            elif name == "cortex_approve":
                text = reader.approve(a["ids"])
            elif name == "cortex_outline":
                text = reader.outline(a["path"])
            elif name == "cortex_stats":
                text = reader.stats()
            else:
                return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        except (OSError, KeyError, ValueError) as e:
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        return {"content": [{"type": "text", "text": text}], "isError": False}
    raise LookupError(m)


def options(env=os.environ):
    """Opt-in extras: CORTEXNET_COPROC=<exported head .npz> (needs torch + transformers for MiniLM),
    CORTEXNET_POLICY=rule_ab, CORTEXNET_GRAPH=1, CORTEXNET_MENU=1 (connected symbols from neighbouring files),
    CORTEXNET_CTX_BASE=<tokens of context before the first read>. COPROC + GRAPH + MENU = the measured full system."""
    kw = dict(policy=env.get("CORTEXNET_POLICY", "budget"), ctx_base=int(env.get("CORTEXNET_CTX_BASE", "20000")))
    if env.get("CORTEXNET_MENU") == "1":
        kw["menu"] = True
    if env.get("CORTEXNET_COPROC"):
        try:
            from .coproc import Coprocessor, MiniLM
            cp = Coprocessor(env["CORTEXNET_COPROC"])        # the head first: a bad path fails before MiniLM
            cp.encoder = MiniLM()
            kw["coproc"] = cp
        except (ImportError, OSError, ValueError) as e:      # ValueError: a head with unusable calibration
            print(f"cortexnet: coprocessor off ({type(e).__name__}: {e})", file=sys.stderr)
    if env.get("CORTEXNET_GRAPH") == "1":
        from .graph import CoNeedGraph
        state = env.get("CORTEXNET_STATE") or os.path.join(os.getcwd(), ".cortexnet")
        path = os.path.join(state, "graph.npz")
        kw["graph"] = CoNeedGraph.load(path) if os.path.exists(path) else CoNeedGraph()
    return kw


def main(stdin=sys.stdin, stdout=sys.stdout):
    reader = Reader(os.environ.get("CORTEXNET_STATE"), split_tokens=int(os.environ.get("CORTEXNET_SPLIT", "600")),
                    **options())
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        if "id" not in req:
            continue        # notification
        try:
            resp = {"jsonrpc": "2.0", "id": req["id"], "result": handle(reader, req)}
        except LookupError as e:
            resp = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": f"method not found: {e}"}}
        except Exception as e:  # keep the server alive; report the failure to the client
            traceback.print_exc(file=sys.stderr)
            resp = {"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32603, "message": str(e)}}
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
