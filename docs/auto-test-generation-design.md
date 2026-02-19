# Auto-Generated Test Suites for VRF
*2026-02-20 06:16 GMT+8*

## Problem

VRF's biggest adoption barrier: **buyers must write test suites**. This is the same friction that limited unit testing adoption until frameworks automated it. Law 39 says every config step halves addressable market. Requiring test suite authorship is a massive config step.

## Insight

The buyer describes what they want in natural language. That description contains implicit test cases:

```
"Write a function that sorts a list of integers in ascending order"
→ Test: sort([3,1,2]) == [1,2,3]
→ Test: sort([]) == []
→ Test: sort([1]) == [1]
→ Test: sort([5,5,5]) == [5,5,5]
```

An LLM can generate these test suites from the task description BEFORE the work is assigned. The test suite becomes part of the contract.

## Design: VRF Test Generation Service

### Flow

```
1. Buyer submits task description (natural language)
2. VRF Test Generator creates test suite (deterministic format)
3. Buyer reviews/approves test suite (optional, can skip)
4. Task + test suite sent to provider
5. Provider submits work
6. VRF verifies work against generated test suite
7. Signed receipt issued
```

### Key Properties

1. **Test generation is subjective, but test execution is deterministic.** The LLM decides WHAT to test. The sandbox decides PASS/FAIL. This cleanly separates concerns.

2. **Tests are visible before work begins.** Provider knows exactly what they'll be judged on. No surprises. No subjective evaluation after the fact.

3. **Buyer can override.** Auto-generated tests are a starting point. Buyer adds/removes tests. This preserves the "test suite as contract" property.

4. **Test quality improves with domain templates.** For common task types (sorting, API endpoints, data transforms), we can maintain curated test templates that the LLM enhances for the specific task.

### Architecture

```
POST /test-gen
{
  "task_description": "Write a Python function that...",
  "language": "python",
  "constraints": ["must handle empty input", "O(n log n)"],
  "template": "function"  // optional: use curated template
}

→ 200 OK
{
  "test_suite": {
    "tests": [...],
    "language": "python",
    "generated_by": "vrf-test-gen/0.1",
    "confidence": 0.85,
    "coverage_estimate": "basic"
  },
  "review_suggested": true
}
```

### Coverage Levels

| Level | Tests | Description |
|-------|-------|-------------|
| basic | 3-5 | Happy path + empty + edge case |
| standard | 8-12 | + type errors, boundary, performance |
| thorough | 20+ | + adversarial, concurrency, security |

### Economics

- Test generation: ~500-1000 tokens (cheap, <$0.01)
- Test execution: existing VRF infrastructure ($0.005/receipt)
- Total: ~$0.015/verified task with auto-generated tests
- Compare: manual test writing = $5-50/task equivalent human time

### Why This Matters

**Without auto-test-gen:**
- Market: developers and sophisticated agents who can write test suites
- Friction: high (must author tests per task)
- Adoption: slow (niche technical users)

**With auto-test-gen:**
- Market: anyone who can describe what they want
- Friction: near-zero (describe task → get verified result)
- Adoption: fast (natural language is universal interface)

### Risks

1. **LLM-generated tests may miss edge cases.** Mitigation: coverage levels, domain templates, buyer review.
2. **Adversarial task descriptions could generate weak tests.** Mitigation: minimum test count, structural requirements.
3. **This reintroduces LLM dependency.** Mitigation: LLM is used for test GENERATION only, not test EXECUTION or JUDGMENT. The separation is clean — generated tests are deterministic artifacts.
4. **Test quality varies by domain.** Mitigation: start with code verification (highest test-generation quality), expand to data/API tasks.

### Relationship to Existing Architecture

- Test gen is a **pre-processing layer**, not a replacement for VRF
- Generated test suites feed into existing `POST /verify` endpoint unchanged
- VRF receipts note `generated_by: vrf-test-gen/0.1` for transparency
- Buyers who provide their own tests skip this entirely

### Implementation Plan

1. **Phase 1: Prompt engineering** — Design prompts that generate valid test suites for common task types
2. **Phase 2: Template library** — Curated test templates for functions, APIs, data transforms, CLI tools
3. **Phase 3: Endpoint** — `POST /test-gen` on verify_server
4. **Phase 4: Integration** — MCP tool `cb_generate_tests`, client SDK, GitHub Action
5. **Phase 5: Feedback loop** — Tests that catch real bugs get promoted; tests that always pass get flagged as weak

### Law 45

**Law 45**: The test suite is the contract. Auto-generating test suites from task descriptions converts natural language intent into deterministic acceptance criteria. This transforms VRF from "verification for developers" to "verification for everyone." The LLM generates the contract; the sandbox enforces it.

---

## Prototype Sketch

```python
def generate_test_suite(task_description: str, language: str = "python",
                        coverage: str = "basic") -> dict:
    """Generate a VRF-compatible test suite from a task description.
    
    Uses an LLM to create deterministic test cases, then validates
    the test suite structure before returning.
    """
    prompt = f"""Generate a test suite for this task:
    
Task: {task_description}
Language: {language}
Coverage: {coverage}

Output format:
{{
  "tests": [
    {{"input": "...", "expected_output": "...", "description": "..."}},
    ...
  ]
}}

Rules:
- Each test must have a concrete input and expected output
- Include: happy path, empty/null input, edge cases
- Tests must be deterministic (no random, no time-dependent)
- Expected outputs must be exact (no "approximately")
"""
    # Call LLM (implementation depends on available API)
    # Validate test suite structure
    # Return VRF-compatible test suite
```

This is a sketch — actual implementation needs:
- LLM API integration (OpenAI key available in 1Password)
- Test suite validation (ensure deterministic, no side effects)
- Language-specific test runners already built in verify_server
