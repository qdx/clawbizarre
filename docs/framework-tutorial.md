# Framework Integration Tutorial

> Add deterministic output verification to any AI agent framework in under 5 minutes.

## Prerequisites

- Python 3.10+
- A running verify_server: `python3 prototype/verify_server_hardened.py`
- That's it. Zero API keys, zero accounts, zero wallets.

---

## 1. Standalone (vrf-client)

The base package. Zero dependencies. Works everywhere.

```python
from vrf_client import VRFClient, extract_code

client = VRFClient()  # defaults to http://localhost:8700

# Your agent generated some code
agent_output = '''
Here's the function:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```
'''

# Extract code from markdown
code = extract_code(agent_output, language="python")

# Verify with a test suite
receipt = client.verify(
    code=code,
    test_suite="assert fibonacci(0) == 0\nassert fibonacci(10) == 55",
    language="python"
)

print(receipt.summary())
# VRF PASS: 2/2 tests (142ms) — a1b2c3d4e5f6

if receipt.passed:
    print("✅ Safe to use this code")
else:
    print(f"❌ Failed: {receipt.failures}")
```

### Environment variable

```bash
export VRF_VERIFY_URL=https://your-server.fly.dev
```

The client picks it up automatically.

---

## 2. LangChain / LangGraph

### As a Tool (agent decides when to verify)

```python
from langchain_vrf import VRFVerifyTool

# Add to your agent's tool list
tools = [VRFVerifyTool()]

# The agent can now call it:
# Action: vrf_verify
# Action Input: {"code": "def add(a,b): return a+b", "test_suite": "assert add(1,2)==3"}
```

### As a Callback (automatic verification on every code block)

```python
from langchain_vrf import VRFVerificationCallback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    callbacks=[VRFVerificationCallback(fail_action="warn")]
    # fail_action: "warn" (log), "raise" (exception), "retry" (ask again)
)

# Every response containing code blocks gets verified automatically
response = llm.invoke("Write a function to sort a list")
# [VRF] Verified code block: PASS 3/3 tests (89ms)
```

### As a LangGraph Node (verification step in your graph)

```python
from langchain_vrf import vrf_verification_node
from langgraph.graph import StateGraph

# Add verification as a graph node
graph = StateGraph(AgentState)
graph.add_node("generate", generate_code)
graph.add_node("verify", vrf_verification_node)
graph.add_node("deliver", deliver_result)

graph.add_edge("generate", "verify")
graph.add_conditional_edges("verify", lambda s: "deliver" if s["vrf_passed"] else "generate")
```

---

## 3. CrewAI

### As a Tool

```python
from crewai_vrf import VRFVerifyTool

coder = Agent(
    role="Python Developer",
    tools=[VRFVerifyTool()],
    goal="Write verified, correct code"
)
```

### As a Task Callback (verify before delivering)

```python
from crewai_vrf import VRFTaskCallback

task = Task(
    description="Write a fibonacci function",
    agent=coder,
    callback=VRFTaskCallback(
        test_suite="assert fibonacci(10) == 55",
        language="python"
    )
)
# Task output is auto-verified. Fails the task if verification fails.
```

### As a Guard (crew-wide verification)

```python
from crewai_vrf import VRFCrewGuard

crew = Crew(
    agents=[coder, reviewer],
    tasks=[coding_task],
    step_callback=VRFCrewGuard(block_on_fail=True)
)
# Every task output with code gets verified before proceeding
```

---

## 4. OpenAI Assistants / Swarm

### As a Function Tool

```python
from openai_vrf import vrf_verify_tool_def, handle_vrf_tool_call

# Register with your assistant
assistant = client.beta.assistants.create(
    model="gpt-4",
    tools=[{"type": "function", "function": vrf_verify_tool_def()}]
)

# In your run loop, handle the tool call:
if tool_call.function.name == "vrf_verify":
    result = handle_vrf_tool_call(tool_call.function.arguments)
```

### As a Swarm Agent

```python
from openai_vrf import create_vrf_agent

verifier = create_vrf_agent()

# Hand off to the verifier from any other agent
def coding_agent_instructions(context):
    return "Write code, then transfer to the Verifier agent for quality check."

coder = Agent(
    name="Coder",
    instructions=coding_agent_instructions,
    functions=[transfer_to_verifier]
)
```

### As a Response Processor (transparent)

```python
from openai_vrf import VRFResponseProcessor

processor = VRFResponseProcessor(auto_verify=True)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a sorting function"}]
)

# Auto-extracts and verifies any code in the response
verified = processor.process(response)
print(verified.vrf_summary)  # "1 code block verified: 1 PASS, 0 FAIL"
```

---

## 5. GitHub Actions (CI/CD)

```yaml
- name: Verify agent output
  uses: clawbizarre/verify-action@v1
  with:
    code: ${{ steps.agent.outputs.code }}
    test_suite: |
      assert sort_list([3,1,2]) == [1,2,3]
      assert sort_list([]) == []
    language: python
```

---

## Patterns

### Verify-then-Use

The simplest pattern. Generate → Verify → Use or Reject.

```python
code = agent.generate(prompt)
receipt = client.verify(code, tests, "python")
if receipt.passed:
    exec(code)  # or deploy, commit, etc.
```

### Verify-Retry Loop

Give the agent its test failures and let it fix the code.

```python
for attempt in range(3):
    code = agent.generate(prompt if attempt == 0 else f"Fix: {receipt.failures}")
    receipt = client.verify(code, tests, "python")
    if receipt.passed:
        break
```

### Staged Verification

Multiple test tiers for progressive confidence.

```python
# Quick smoke test
receipt = client.verify(code, smoke_tests, "python")
if receipt.passed:
    # Full test suite
    receipt = client.verify(code, full_tests, "python")
    if receipt.passed:
        # Integration tests (Docker sandboxed)
        receipt = client.verify(code, integration_tests, "python", use_docker=True)
```

### Receipt Chain (Audit Trail)

Every verification produces a receipt. Chain them for compliance.

```python
receipts = []
for task in tasks:
    code = agent.do(task)
    receipt = client.verify(code, task.tests, task.language)
    receipts.append(receipt.raw)

# receipts[] is your audit trail — tamper-evident, timestamped, deterministic
```

---

## What Gets Verified

VRF verifies **functional correctness** — does the code produce the right output?

| ✅ VRF catches | ❌ VRF doesn't catch |
|---|---|
| Wrong logic | Style issues |
| Off-by-one errors | Performance problems |
| Missing edge cases | Security vulnerabilities (unless tested) |
| Type errors | Maintainability |
| Runtime exceptions | Design quality |

Write better test suites → catch more problems. The verification is only as good as the tests.

---

## FAQ

**Q: What if verify_server is down?**
All integrations degrade gracefully. VRFClient raises `VRFError`; callbacks log warnings; guards pass through. No verification is treated as "unverified", never as "passed."

**Q: Can I run verify_server in Docker?**
```bash
docker run -p 8700:8700 clawbizarre/verify-server
```

**Q: What languages are supported?**
Python (native), JavaScript and Bash (via Docker). More coming.

**Q: How fast is verification?**
Typical: 50-200ms per verification. Dominated by code execution time, not protocol overhead.

**Q: Is it safe to execute untrusted code?**
verify_server uses subprocess isolation with timeouts. For untrusted code, use `use_docker=True` for full container sandboxing (network disabled, memory limited).
