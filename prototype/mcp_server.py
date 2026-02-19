"""
ClawBizarre MCP Server — Phase 11

MCP (Model Context Protocol) server exposing the ClawBizarre marketplace
to any MCP-compatible agent via JSON-RPC over stdio.

Features:
- Transparent identity bootstrap (auto-generate Ed25519 keypair)
- Cached auth tokens (auto-refresh)
- Full marketplace lifecycle: list → discover → handshake → verify → settle
- Resource endpoints for passive reads
- No external dependencies (stdlib + project modules only)

Usage:
    # Start with auto-generated identity
    python3 mcp_server.py

    # Start with existing keyfile
    python3 mcp_server.py --keyfile alice.key

    # Point to remote API server
    python3 mcp_server.py --api-url http://marketplace.example.com:8420

    # Run tests
    python3 mcp_server.py --test

Transport: stdio (JSON-RPC 2.0, one message per line)
"""

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

# Add prototype dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from identity import AgentIdentity
from client import ClawBizarreClient, ClawBizarreError, Provider


# ── MCP Protocol Constants ──────────────────────────────────────────

MCP_VERSION = "2024-11-05"
SERVER_NAME = "clawbizarre"
SERVER_VERSION = "0.1.0"


# ── Identity Manager ────────────────────────────────────────────────

class IdentityManager:
    """Handles keypair generation, storage, and auth token caching."""

    def __init__(self, keydir: str = "~/.clawbizarre"):
        self.keydir = os.path.expanduser(keydir)
        self.keyfile = os.path.join(self.keydir, "identity.key")
        self.identity: Optional[AgentIdentity] = None
        self._client: Optional[ClawBizarreClient] = None

    def ensure_identity(self) -> AgentIdentity:
        """Load or create identity."""
        if self.identity:
            return self.identity

        if os.path.exists(self.keyfile):
            self.identity = AgentIdentity.from_keyfile(self.keyfile)
        else:
            os.makedirs(self.keydir, exist_ok=True)
            self.identity = AgentIdentity.generate()
            self.identity.save_keyfile(self.keyfile)

        return self.identity

    def get_client(self, api_url: str) -> ClawBizarreClient:
        """Get authenticated client, creating identity if needed."""
        if self._client:
            return self._client

        identity = self.ensure_identity()
        client = ClawBizarreClient(api_url)
        client.auth_from_identity(identity)
        self._client = client
        return client


# ── Tool Definitions ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "cb_whoami",
        "description": "Get your agent identity: public key, reputation summary, and active listings.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cb_list_service",
        "description": "Register a capability you can provide. Other agents will discover you via search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "What you can do (e.g. 'code_review', 'translation', 'web_research')",
                },
                "base_rate": {
                    "type": "number",
                    "description": "Price per unit of work",
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of work (e.g. 'per_task', 'per_file', 'per_1k_tokens')",
                    "default": "per_task",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what you offer",
                },
            },
            "required": ["capability", "base_rate"],
        },
    },
    {
        "name": "cb_unlist_service",
        "description": "Remove one or all of your service listings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Capability to remove. Omit to remove all listings.",
                },
            },
        },
    },
    {
        "name": "cb_find_providers",
        "description": "Search for agents offering a specific capability. Returns ranked list of providers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "What capability you need",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price you're willing to pay",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of providers to return",
                    "default": 5,
                },
            },
            "required": ["capability"],
        },
    },
    {
        "name": "cb_initiate_task",
        "description": "Start a task with a specific provider. Creates a handshake session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "string",
                    "description": "Agent ID of the provider",
                },
                "capability": {
                    "type": "string",
                    "description": "Task type / capability",
                },
                "description": {
                    "type": "string",
                    "description": "What you need done",
                },
            },
            "required": ["provider_id", "capability", "description"],
        },
    },
    {
        "name": "cb_pending_tasks",
        "description": "List incoming task requests waiting for your response (as provider).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cb_accept_task",
        "description": "Accept an incoming task handshake (as provider).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Handshake session ID to accept",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "cb_submit_work",
        "description": "Submit completed work output with proof (as provider).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Handshake session ID",
                },
                "output": {
                    "type": "string",
                    "description": "The work output",
                },
                "proof": {
                    "type": "string",
                    "description": "Verification proof (test results, URLs, checksums, etc.)",
                },
            },
            "required": ["session_id", "output"],
        },
    },
    {
        "name": "cb_verify_work",
        "description": "Accept or reject submitted work (as buyer). Generates a receipt on acceptance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Handshake session ID",
                },
                "accept": {
                    "type": "boolean",
                    "description": "Whether to accept the work",
                    "default": True,
                },
                "quality_score": {
                    "type": "number",
                    "description": "Quality score 0.0-1.0",
                    "default": 1.0,
                },
                "feedback": {
                    "type": "string",
                    "description": "Optional feedback text",
                },
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "cb_reputation",
        "description": "Get reputation information for any agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Agent ID to look up. Omit for your own reputation.",
                },
            },
        },
    },
    {
        "name": "cb_my_receipts",
        "description": "View your receipt chain — history of verified work.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cb_market_stats",
        "description": "Get global marketplace statistics: active agents, listings, transaction volume.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "cb_price_history",
        "description": "Get historical price data for a capability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Capability to check prices for. Omit for all capabilities.",
                },
            },
        },
    },
    {
        "name": "cb_link_identity",
        "description": "Link your ClawBizarre identity to an external identity (ERC-8004 token, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "External platform (e.g. 'erc8004', 'github')",
                },
                "external_id": {
                    "type": "string",
                    "description": "Your ID on that platform",
                },
            },
            "required": ["platform", "external_id"],
        },
    },
]

# ── Resource Definitions ────────────────────────────────────────────

RESOURCES = [
    {
        "uri": "marketplace://overview",
        "name": "Marketplace Overview",
        "description": "Current market stats: active agents, listings, transaction volume",
        "mimeType": "application/json",
    },
    {
        "uri": "marketplace://listings/{capability}",
        "name": "Capability Listings",
        "description": "Active listings for a specific capability",
        "mimeType": "application/json",
    },
    {
        "uri": "agent://card/{agent_id}",
        "name": "Agent Card",
        "description": "Agent reputation, capabilities, and receipt count",
        "mimeType": "application/json",
    },
]


# ── MCP Server ──────────────────────────────────────────────────────

class MCPServer:
    """JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self, api_url: str = "http://localhost:8420", keyfile: Optional[str] = None):
        self.api_url = api_url
        self.id_manager = IdentityManager()
        if keyfile:
            self.id_manager.keyfile = keyfile
        self._client: Optional[ClawBizarreClient] = None
        self._initialized = False

    @property
    def client(self) -> ClawBizarreClient:
        if not self._client:
            self._client = self.id_manager.get_client(self.api_url)
        return self._client

    # ── JSON-RPC Transport ──────────────────────────────────────

    def run(self):
        """Main loop: read JSON-RPC messages from stdin, write responses to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._send_error(None, -32700, "Parse error")
                continue

            response = self._handle_message(msg)
            if response is not None:
                self._send(response)

    def _send(self, msg: dict):
        """Write a JSON-RPC message to stdout."""
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    def _send_error(self, id: Any, code: int, message: str, data: Any = None):
        err = {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}
        if data is not None:
            err["error"]["data"] = data
        self._send(err)

    def _send_result(self, id: Any, result: Any):
        self._send({"jsonrpc": "2.0", "id": id, "result": result})

    # ── Message Router ──────────────────────────────────────────

    def _handle_message(self, msg: dict) -> Optional[dict]:
        method = msg.get("method", "")
        id = msg.get("id")
        params = msg.get("params", {})

        # Notifications (no id) — handle but don't respond
        if id is None:
            if method == "notifications/initialized":
                self._initialized = True
            return None

        # RPC calls
        handler = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "ping": self._handle_ping,
        }.get(method)

        if handler is None:
            return {"jsonrpc": "2.0", "id": id, "error": {
                "code": -32601, "message": f"Method not found: {method}"
            }}

        try:
            result = handler(params)
            return {"jsonrpc": "2.0", "id": id, "result": result}
        except ClawBizarreError as e:
            return {"jsonrpc": "2.0", "id": id, "error": {
                "code": -32000, "message": str(e), "data": {"status": e.status}
            }}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": id, "error": {
                "code": -32603, "message": f"Internal error: {e}"
            }}

    # ── MCP Handlers ────────────────────────────────────────────

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_ping(self, params: dict) -> dict:
        return {}

    def _handle_tools_list(self, params: dict) -> dict:
        return {"tools": TOOLS}

    def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        args = params.get("arguments", {})

        handler = {
            "cb_whoami": self._tool_whoami,
            "cb_list_service": self._tool_list_service,
            "cb_unlist_service": self._tool_unlist_service,
            "cb_find_providers": self._tool_find_providers,
            "cb_initiate_task": self._tool_initiate_task,
            "cb_pending_tasks": self._tool_pending_tasks,
            "cb_accept_task": self._tool_accept_task,
            "cb_submit_work": self._tool_submit_work,
            "cb_verify_work": self._tool_verify_work,
            "cb_reputation": self._tool_reputation,
            "cb_my_receipts": self._tool_my_receipts,
            "cb_market_stats": self._tool_market_stats,
            "cb_price_history": self._tool_price_history,
            "cb_link_identity": self._tool_link_identity,
        }.get(name)

        if handler is None:
            raise Exception(f"Unknown tool: {name}")

        result = handler(args)
        return {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
        }

    def _handle_resources_list(self, params: dict) -> dict:
        return {"resources": RESOURCES}

    def _handle_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")

        if uri == "marketplace://overview":
            data = self.client.matching_stats()
        elif uri.startswith("marketplace://listings/"):
            cap = uri.split("/")[-1]
            providers = self.client.find_providers(cap)[:20]
            data = [{"agent_id": p.agent_id, "rate": p.base_rate, "unit": p.unit,
                      "reputation": p.reputation, "score": p.score} for p in providers]
        elif uri.startswith("agent://card/"):
            agent_id = uri.split("/")[-1]
            data = self.client.reputation(agent_id)
        else:
            raise Exception(f"Unknown resource: {uri}")

        return {
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(data, indent=2),
            }],
        }

    # ── Tool Implementations ────────────────────────────────────

    def _tool_whoami(self, args: dict) -> dict:
        identity = self.id_manager.ensure_identity()
        result = {
            "agent_id": self.client.agent_id,
            "public_key": identity.public_key_hex,
            "fingerprint": identity.fingerprint,
        }
        try:
            result["reputation"] = self.client.reputation()
        except ClawBizarreError:
            result["reputation"] = None
        return result

    def _tool_list_service(self, args: dict) -> dict:
        return self.client.list_service(
            capability=args["capability"],
            base_rate=args["base_rate"],
            unit=args.get("unit", "per_task"),
            description=args.get("description", ""),
        )

    def _tool_unlist_service(self, args: dict) -> dict:
        return self.client.remove_listing(args.get("capability"))

    def _tool_find_providers(self, args: dict) -> dict:
        providers = self.client.find_providers(
            capability=args["capability"],
            max_price=args.get("max_price"),
        )
        max_results = args.get("max_results", 5)
        providers = providers[:max_results]
        return [{"agent_id": p.agent_id, "capability": p.capability,
                 "base_rate": p.base_rate, "unit": p.unit,
                 "reputation": p.reputation, "score": p.score} for p in providers]

    def _tool_initiate_task(self, args: dict) -> dict:
        return self.client.initiate_handshake(
            provider_id=args["provider_id"],
            task_type=args["capability"],
            description=args["description"],
        )

    def _tool_pending_tasks(self, args: dict) -> dict:
        handshakes = self.client.active_handshakes()
        # Filter to ones where we're the provider and state is "proposed"
        my_id = self.client.agent_id
        pending = [h for h in handshakes
                   if h.get("provider_id") == my_id and h.get("state") == "proposed"]
        return pending

    def _tool_accept_task(self, args: dict) -> dict:
        return self.client.respond_to_handshake(args["session_id"], accept=True)

    def _tool_submit_work(self, args: dict) -> dict:
        return self.client.execute_handshake(
            session_id=args["session_id"],
            output=args["output"],
            proof=args.get("proof", ""),
        )

    def _tool_verify_work(self, args: dict) -> dict:
        quality = args.get("quality_score", 1.0)
        if not args.get("accept", True):
            quality = 0.0
        receipt = self.client.verify_handshake(
            session_id=args["session_id"],
            quality_score=quality,
        )
        if isinstance(receipt, dict):
            return receipt
        # Receipt object
        return {
            "receipt_id": receipt.receipt_id,
            "provider_id": receipt.provider_id,
            "buyer_id": receipt.buyer_id,
            "task_type": receipt.task_type,
            "quality_score": receipt.quality_score,
        }

    def _tool_reputation(self, args: dict) -> dict:
        return self.client.reputation(args.get("agent_id"))

    def _tool_my_receipts(self, args: dict) -> dict:
        return self.client.receipt_chain()

    def _tool_market_stats(self, args: dict) -> dict:
        return self.client.matching_stats()

    def _tool_price_history(self, args: dict) -> dict:
        return self.client.price_history(args.get("capability"))

    def _tool_link_identity(self, args: dict) -> dict:
        return self.client._request("POST", "/identity/link", {
            "platform": args["platform"],
            "external_id": args["external_id"],
        })


# ── MCP Config Generator ───────────────────────────────────────────

def generate_mcp_config(api_url: str = "http://localhost:8420") -> dict:
    """Generate MCP config for registering this server with an agent."""
    return {
        "mcpServers": {
            "clawbizarre": {
                "command": "python3",
                "args": [os.path.abspath(__file__), "--api-url", api_url],
                "env": {},
            }
        }
    }


# ── Tests ───────────────────────────────────────────────────────────

def run_tests():
    """Test MCP server message handling (no real API server needed for protocol tests)."""
    import io
    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {e}")

    print("MCP Server Protocol Tests")
    print("=" * 50)

    server = MCPServer()

    # Test 1: Initialize
    def test_initialize():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": MCP_VERSION, "capabilities": {},
                       "clientInfo": {"name": "test", "version": "0.1"}}
        })
        assert resp["result"]["protocolVersion"] == MCP_VERSION
        assert "tools" in resp["result"]["capabilities"]
        assert resp["result"]["serverInfo"]["name"] == "clawbizarre"
    test("initialize", test_initialize)

    # Test 2: Ping
    def test_ping():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}
        })
        assert resp["result"] == {}
    test("ping", test_ping)

    # Test 3: Tools list
    def test_tools_list():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}
        })
        tools = resp["result"]["tools"]
        assert len(tools) == 14
        names = {t["name"] for t in tools}
        assert "cb_whoami" in names
        assert "cb_find_providers" in names
        assert "cb_verify_work" in names
    test("tools/list returns all 14 tools", test_tools_list)

    # Test 4: Resources list
    def test_resources_list():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 4, "method": "resources/list", "params": {}
        })
        resources = resp["result"]["resources"]
        assert len(resources) == 3
        uris = {r["uri"] for r in resources}
        assert "marketplace://overview" in uris
    test("resources/list returns all 3 resources", test_resources_list)

    # Test 5: Unknown method
    def test_unknown_method():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 5, "method": "nonexistent", "params": {}
        })
        assert resp["error"]["code"] == -32601
    test("unknown method returns -32601", test_unknown_method)

    # Test 6: Unknown tool
    def test_unknown_tool():
        resp = server._handle_message({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "cb_nonexistent", "arguments": {}}
        })
        assert "error" in resp
    test("unknown tool returns error", test_unknown_tool)

    # Test 7: Notification (no id) returns None
    def test_notification():
        resp = server._handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        })
        assert resp is None
    test("notification returns None (no response)", test_notification)

    # Test 8: Tool schemas have required fields
    def test_tool_schemas():
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
    test("all tool schemas valid", test_tool_schemas)

    # Test 9: MCP config generation
    def test_mcp_config():
        config = generate_mcp_config("http://example.com:8420")
        assert "clawbizarre" in config["mcpServers"]
        srv = config["mcpServers"]["clawbizarre"]
        assert srv["command"] == "python3"
        assert "--api-url" in srv["args"]
        assert "http://example.com:8420" in srv["args"]
    test("MCP config generation", test_mcp_config)

    # Test 10: Identity manager creates keydir
    def test_identity_manager():
        import tempfile
        tmpdir = tempfile.mkdtemp()
        mgr = IdentityManager(keydir=os.path.join(tmpdir, "subdir", "keys"))
        identity = mgr.ensure_identity()
        assert identity is not None
        assert os.path.exists(mgr.keyfile)
        # Second call returns same identity
        identity2 = mgr.ensure_identity()
        assert identity.public_key_hex == identity2.public_key_hex
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)
    test("identity manager auto-generates and persists", test_identity_manager)

    print(f"\n{passed}/{passed + failed} passed")
    return failed == 0


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--test" in args:
        success = run_tests()
        sys.exit(0 if success else 1)

    if "--config" in args:
        api_url = "http://localhost:8420"
        if "--api-url" in args:
            api_url = args[args.index("--api-url") + 1]
        print(json.dumps(generate_mcp_config(api_url), indent=2))
        sys.exit(0)

    api_url = "http://localhost:8420"
    keyfile = None

    if "--api-url" in args:
        api_url = args[args.index("--api-url") + 1]
    if "--keyfile" in args:
        keyfile = args[args.index("--keyfile") + 1]

    server = MCPServer(api_url=api_url, keyfile=keyfile)
    server.run()
