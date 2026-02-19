# Contributing to ClawBizarre

Thanks for your interest! ClawBizarre is an open-source verification protocol for AI agent work — the "SSL certificate for agent output."

## Quick Start

```bash
git clone https://github.com/qdx/clawbizarre.git
cd clawbizarre

# Start the verification server (no dependencies beyond Python 3.10+)
python3 prototype/verify_server_unified.py

# Verify your first piece of code
curl -X POST http://localhost:8700/verify \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b): return a + b",
    "test_suite": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
    "language": "python"
  }'
```

## Project Structure

```
prototype/           # Core implementation (Python, stdlib only)
  verify_server_unified.py   # Main server (verification + transparency)
  verify_server_hardened.py  # Standalone verification server
  identity.py               # Ed25519 agent identity
  receipt.py                # VRF receipt format
  vrf_cose.py               # COSE/SCITT encoding
  merkle.py                 # RFC 9162 Merkle tree
  mcp_server.py             # MCP distribution (14 tools)
  a2a_adapter.py            # Google A2A adapter
  acp_evaluator.py          # Virtuals ACP evaluator bridge
  client.py                 # Python SDK
  integrations/             # Framework integrations
    vrf_client.py           # Shared client library
    langchain_vrf.py        # LangChain/LangGraph
    crewai_vrf.py           # CrewAI
    openai_vrf.py           # OpenAI Assistants/Swarm
packages/                   # PyPI package structures
github-action/              # GitHub Action for CI/CD
acp-deploy/                 # ACP deployment package
skills/clawbizarre/         # OpenClaw skill
```

## How to Contribute

### Good First Issues

- **Add a new language runtime** — `verify_server` supports Python, JavaScript, and Bash. Add Ruby, Go, or Rust via `docker_runner.py`.
- **Write framework integrations** — AutoGen, Microsoft Agent Framework, or Semantic Kernel wrappers.
- **Improve test suites** — More edge cases for receipt verification, chain linking, Merkle proofs.
- **Documentation** — Tutorials, examples, diagrams.

### Design Contributions

We welcome research and design work:
- Economic modeling (extend simulations v1-v10)
- Protocol spec review (VRF spec v1.0, SCITT alignment)
- Security analysis (sandbox escape vectors, replay attacks)

### Code Standards

- **Zero external dependencies** for core protocol (stdlib only). Framework integrations may depend on their respective frameworks.
- **Every module includes inline tests** — run with `python3 <module>.py`. Tests are at the bottom of each file.
- **Type hints** on all public functions.
- **Docstrings** on all public classes and functions.

### Testing

```bash
# Run all tests (each file is self-contained)
for f in prototype/*.py; do python3 "$f" 2>/dev/null; done

# Run specific module tests
python3 prototype/verify_server_unified.py  # 27 tests
python3 prototype/mcp_server.py             # 10 tests
python3 prototype/integrations/vrf_client.py # 13 tests
```

### Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-thing`)
3. Add/update tests (inline at bottom of module)
4. Ensure all module tests pass
5. Submit PR with description of what and why

## Architecture Decisions

Key design choices documented in:
- `design-document-v2.md` — System consolidation, proven vs speculative
- `verification-protocol-v1.md` — Protocol design rationale
- `vrf-spec-v1.md` — Receipt format specification
- `matching-engine-design.md` — Why posted-price over auction
- `scitt-alignment.md` — IETF standards mapping
- `adoption-playbook.md` — Adoption strategy

## Communication

- **Issues**: GitHub Issues for bugs, feature requests, design discussions
- **PRs**: Always welcome, even small ones
- **Design proposals**: Open an issue first for anything architectural

## License

MIT — use it however you want.
