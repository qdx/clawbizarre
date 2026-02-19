# VRF Framework Integrations Design

## Context (2026-02-20)
The agent framework landscape has consolidated around 5 major players:
1. **LangChain/LangGraph** — largest ecosystem, graph-based control (Law 40 integration designed)
2. **CrewAI** — role-based, pragmatic, but 44% failure rate in benchmarks (DEV Community, Feb 2026)
3. **Microsoft Agent Framework (MAF)** — unified SK + AutoGen (Oct 2025), .NET-first, preview
4. **OpenAI Swarm/Assistants** — lowest barrier, black-box state, vendor lock-in ceiling
5. **AutoGen (legacy)** — declining interest, being absorbed into MAF

### Key Finding: None Have Deterministic Output Verification
- CrewAI: "human-in-the-loop verification at critical decision points" — manual, not structural
- AutoGen: "Conversational Chaos" — known infinite loop / token bleeding problem. No output checks.
- MAF: OpenTelemetry observability, filters/middleware — monitoring, not verification
- OpenAI: "Lack of Determinism" explicitly called out as migration reason. Black-box threads.
- LangGraph: Most controllable (explicit graph), but verification = "add a node" (no standard)

**Law 41**: Every major agent framework solves orchestration. None solve output verification. The verification gap is framework-universal, not framework-specific. A single verification protocol serves all frameworks.

## Integration Priority (by addressable market)

### Tier 1: Python (80%+ of agent developers)
1. **`langchain-vrf`** — LangChain/LangGraph (designed in langchain-integration.md)
2. **`crewai-vrf`** — CrewAI integration (new)
3. **`openai-vrf`** — OpenAI Assistants/Swarm (new)

### Tier 2: .NET (enterprise, growing)
4. **`Microsoft.Agents.VRF`** — MAF NuGet package (new)

### Tier 3: Framework-agnostic
5. **HTTP API** — already built (verify_server)
6. **MCP Server** — already built (Phase 11)
7. **GitHub Action** — already built

## CrewAI Integration Design (`crewai-vrf`)

### CrewAI Architecture
- **Agents** have roles, goals, backstories
- **Tasks** have descriptions, expected outputs, tools
- **Crews** orchestrate agents executing tasks
- **Flows** (newer): structured state machines with `@listen` decorators

### Integration Points

#### Level 1: Tool (agent-initiated)
```python
from crewai_vrf import VerifyCodeTool

# Add to any agent's tool list
coder = Agent(
    role="Senior Developer",
    tools=[VerifyCodeTool(verify_url="http://localhost:8700")]
)

# Agent decides when to verify its own output
task = Task(
    description="Write a function to sort a list. Verify it works before submitting.",
    agent=coder
)
```

#### Level 2: Task Callback (automatic)
```python
from crewai_vrf import vrf_task_callback

# Auto-verify any task that produces code
task = Task(
    description="Write a fibonacci function",
    expected_output="Working Python code",
    callback=vrf_task_callback(
        verify_url="http://localhost:8700",
        language="python",
        test_suite="assert fibonacci(10) == 55"
    )
)
```

#### Level 3: Crew-level Guard (structural)
```python
from crewai_vrf import VRFGuardedCrew

# All code-producing tasks auto-verified, retry on failure
crew = VRFGuardedCrew(
    agents=[coder, reviewer],
    tasks=[coding_task],
    verify_url="http://localhost:8700",
    max_retries=2
)
```

### CrewAI-Specific Value Prop
CrewAI's 44% failure rate in benchmarks = strong need for output verification.
VRF catches failures that CrewAI's role-based "review" pattern misses (LLM reviewing LLM ≠ deterministic).

## OpenAI Assistants/Swarm Integration Design (`openai-vrf`)

### OpenAI Architecture
- **Assistants API**: Thread-based, tool calling, code interpreter
- **Swarm**: Lightweight hand-off between agents, function-based routing

### Integration Points

#### Level 1: Function Tool
```python
from openai_vrf import verify_code_tool

# Register as OpenAI function tool
tools = [verify_code_tool(verify_url="http://localhost:8700")]

assistant = client.beta.assistants.create(
    name="Verified Coder",
    instructions="Always verify code before returning it.",
    tools=tools,
    model="gpt-4o"
)
```

#### Level 2: Swarm Verification Agent
```python
from openai_vrf import make_verification_agent

# Dedicated verification agent in Swarm hand-off chain
verifier = make_verification_agent(verify_url="http://localhost:8700")

# Coder → Verifier → User (hand-off chain)
coder_agent = Agent(
    name="Coder",
    functions=[hand_off_to_verifier]
)
```

#### Level 3: Run Step Hook
```python
from openai_vrf import VRFRunStepHook

# Intercept assistant run steps, verify code outputs
hook = VRFRunStepHook(
    verify_url="http://localhost:8700",
    auto_retry=True
)
```

## Microsoft Agent Framework Integration Design (`Microsoft.Agents.VRF`)

### MAF Architecture
- **ChatClientAgent**: Single agent with tools
- **GroupChat**: Multi-agent orchestration (round-robin, selector, broadcast)
- **Workflows**: Durable, checkpointed task sequences
- **Filters/Middleware**: Pipeline-level interceptors

### Integration Points

#### Level 1: Agent Tool
```csharp
public class VrfTools
{
    private readonly VrfClient _client;
    
    [AgentTool]
    [Description("Verify code output against a test suite")]
    public async Task<VrfReceipt> VerifyCode(
        [Description("The code to verify")] string code,
        [Description("Test assertions")] string testSuite,
        [Description("Programming language")] string language = "python")
    {
        return await _client.VerifyAsync(code, testSuite, language);
    }
}
```

#### Level 2: Middleware Filter
```csharp
// Auto-verify all tool outputs that look like code
services.AddSingleton<IAgentFilter>(new VrfVerificationFilter(
    verifyUrl: "http://localhost:8700",
    languages: ["python", "javascript", "csharp"]
));
```

#### Level 3: Workflow Node
```csharp
// In MAF Workflow: code → verify → branch (pass/retry)
workflow.AddStep("generate_code", codeAgent)
    .Then("verify", new VrfVerificationStep())
    .OnPass("deliver")
    .OnFail("retry", maxAttempts: 2);
```

### MAF-Specific Value Prop
Enterprise .NET developers are MAF's audience. They need compliance + audit trails.
VRF receipts = exactly the "deterministic validation" that drives migration from OpenAI (per benchmark article).
OpenTelemetry integration: VRF receipts as trace spans.

## Package Distribution Plan

| Package | Registry | Language | Dependencies |
|---------|----------|----------|-------------|
| `langchain-vrf` | PyPI | Python | langchain-core |
| `crewai-vrf` | PyPI | Python | crewai |
| `openai-vrf` | PyPI | Python | openai |
| `Microsoft.Agents.VRF` | NuGet | C# | Microsoft.Agents.AI |
| `vrf-client` | PyPI | Python | none (stdlib) — already built as client.py |

All packages depend on a running verify_server instance (or hosted endpoint).
No packages require crypto wallets, blockchain, or accounts.

## Implementation Priority
1. **`vrf-client`** — rename/publish existing client.py (zero deps, works with any framework)
2. **`crewai-vrf`** — highest pain point (44% failure rate), thin wrapper over vrf-client
3. **`openai-vrf`** — largest user base, function tool format
4. **`langchain-vrf`** — 3-level design already done
5. **`Microsoft.Agents.VRF`** — .NET, longer timeline, enterprise audience

## Shared Pattern
All integrations follow the same 3-level depth (Law 40):
1. **Tool** — agent calls verification when it wants to (opt-in)
2. **Callback/Middleware** — automatic verification on code outputs (passive)
3. **Structural** — verification as workflow gate with retry (enforced)

Each level captures a different adoption stage: experimentation → integration → production.
