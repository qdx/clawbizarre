# ClawBizarre ACP Service Offering Design
*2026-02-19 10:31 GMT+8*

## Service: Structural Code Verification

### Phase 1 Answers (from seller reference template)

**1. What does the job do?**
- Name: `structural_code_verification`
- Description: "Structural code verification with test suites. Submit code + tests, get a cryptographically signed verification receipt (VRF). Supports Python, JavaScript, Bash. Docker-sandboxed execution. Deterministic pass/fail — no LLM subjectivity."
- Problem it solves: Buyers/evaluators need to know if delivered code actually works. LLM-subjective evaluation is unreliable. We run the tests and produce a signed receipt.

**2. Existing functionality?**
- Yes: `verify_server.py` — full HTTP verification server
  - Tier 0: test suite execution (Python/JS/Bash, Docker sandbox)
  - Tier 1: schema/constraint validation
  - VRF receipt generation with Ed25519 signatures
  - 37/37 tests passing

**3. Job inputs/requirements?**
```json
{
  "type": "object",
  "required": ["code", "test_suite"],
  "properties": {
    "code": {
      "type": "string",
      "description": "The code to verify"
    },
    "test_suite": {
      "type": "string", 
      "description": "Test code that exercises the submitted code. Must use assert statements."
    },
    "language": {
      "type": "string",
      "enum": ["python", "javascript", "bash"],
      "default": "python",
      "description": "Programming language of the code and tests"
    },
    "use_docker": {
      "type": "boolean",
      "default": false,
      "description": "Force Docker sandboxing (network-isolated, memory-limited)"
    }
  }
}
```

**4. Fee / business model?**
- Type: `fixed`
- Fee: $0.005 USDC per verification (validated by simulation — Law 16: invisible below 1% of task value)
- No additional funds needed

**5. Additional funds?**
- No (`requiredFunds: false`)

**6. Execution logic:**
1. Receive job with code + test_suite + language
2. POST to verify_server `/verify` endpoint
3. verify_server executes tests in sandbox
4. Return VRF receipt as deliverable (JSON with pass/fail, test results, signed receipt)
5. If verify_server is down, reject job (don't deliver unverified results)

**7. Returns funds?**
- No

**8. Validation?**
- Reject if code is empty
- Reject if test_suite is empty
- Reject if language not in supported list
- Reject if code exceeds 50KB (safety limit)

### offering.json (draft)
```json
{
  "name": "structural_code_verification",
  "description": "Structural code verification with test suites. Submit code + tests, get a cryptographically signed VRF receipt. Supports Python, JavaScript, Bash. Docker-sandboxed. Deterministic — no LLM subjectivity.",
  "jobFeeType": "fixed",
  "jobFee": 0.005,
  "requiredFunds": false,
  "requirement": {
    "type": "object",
    "required": ["code", "test_suite"],
    "properties": {
      "code": { "type": "string" },
      "test_suite": { "type": "string" },
      "language": { "type": "string", "enum": ["python", "javascript", "bash"], "default": "python" },
      "use_docker": { "type": "boolean", "default": false }
    }
  }
}
```

### handlers.ts (pseudocode)
```typescript
async function executeJob(job: Job): Promise<JobResult> {
  const { code, test_suite, language, use_docker } = job.requirements;
  
  const response = await fetch(`${VERIFY_SERVER_URL}/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      code,
      verification: {
        test_suite: { code: test_suite, language: language || 'python' },
        use_docker: use_docker || false
      }
    })
  });
  
  if (!response.ok) {
    throw new Error(`Verification server error: ${response.status}`);
  }
  
  const receipt = await response.json();
  
  return {
    deliverable: JSON.stringify({
      verification_result: receipt.tier_0_passed ? 'PASS' : 'FAIL',
      tests_passed: receipt.test_results?.passed || 0,
      tests_failed: receipt.test_results?.failed || 0,
      vrf_receipt: receipt,
      verified_at: new Date().toISOString()
    })
  };
}

async function validateRequirements(requirements: any): Promise<boolean> {
  if (!requirements.code || requirements.code.trim().length === 0) return false;
  if (!requirements.test_suite || requirements.test_suite.trim().length === 0) return false;
  if (requirements.code.length > 50000) return false;
  if (requirements.language && !['python', 'javascript', 'bash'].includes(requirements.language)) return false;
  return true;
}
```

### Deployment Dependencies
1. verify_server deployed publicly (Fly.io or EC2)
2. openclaw-acp CLI installed (`npm install`)
3. `acp setup` run (creates wallet + agent)
4. Offering registered + serve started

### Pricing Rationale
- $0.005/verification = $0.50 per 100 verifications
- Break-even at ~100 verifications/day on Fly.io free tier
- Simulation validated: agents insensitive to fees <1% of task value (Law 16)
- 15% platform fee ceiling means ACP takes ~$0.00075/job at most
