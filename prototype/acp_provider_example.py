#!/usr/bin/env python3
"""
Example: ACP Provider Agent with ClawBizarre Pre-Verification

Shows the complete flow of an ACP provider agent that:
1. Receives a job request
2. Does the work (code generation)
3. Pre-verifies via ClawBizarre
4. Delivers verified work to buyer

This is a standalone example — no ACP SDK or verify_server needed.
It demonstrates the PATTERN that real agents would follow.
"""

from provider_verify import (
    ProviderVerifyClient, VerificationRequest,
    VerificationResult, format_acp_delivery_memo
)
import json


class MockACPProvider:
    """
    Simulates an ACP provider agent's workflow.
    
    In production, this would use virtuals-acp SDK:
        from virtuals_acp.client import VirtualsACP
        acp = VirtualsACP(...)
    """

    def __init__(self, verify_url="http://localhost:9800"):
        self.verifier = ProviderVerifyClient(verify_url)
        self.agent_name = "CodeCraft-Agent"
        self.agent_address = "0xCodeCraft..."

    def on_new_task(self, job):
        """
        Called when buyer initiates a job.
        
        ACP flow: buyer calls initiate_job → provider receives via on_new_task callback
        """
        print(f"\n📥 New job received: {job['id']}")
        print(f"   Buyer: {job['buyer']}")
        print(f"   Requirement: {job['requirement']}")

        # Step 1: Accept the job
        print(f"\n✅ Accepting job {job['id']}")
        # acp.respond_job(job['id'], memo_id, accept=True, reason="Can do this")

        # Step 2: Do the work
        print(f"\n🔨 Working on deliverable...")
        code = self._generate_code(job['requirement'])

        # Step 3: Pre-verify with ClawBizarre
        print(f"\n🔍 Pre-verifying with ClawBizarre...")
        verification = self._pre_verify(code, job)

        # Step 4: Deliver (with or without verification)
        self._deliver(job, code, verification)

    def _generate_code(self, requirement: str) -> str:
        """Generate code for the requirement. (Mock — real agent uses LLM)"""
        # In production: call LLM to generate code
        return '''
def sort_and_deduplicate(items):
    """Sort a list and remove duplicates, preserving order of first occurrence."""
    seen = set()
    result = []
    for item in sorted(items):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
'''

    def _pre_verify(self, code: str, job: dict) -> VerificationResult:
        """Pre-verify code before delivering to buyer."""

        # Extract test suite from job requirements if present
        test_suite = job.get("test_suite")
        if not test_suite:
            # Generate basic tests from requirement
            test_suite = [
                {"input": "sort_and_deduplicate([3,1,2,1,3])", "expected": "[1, 2, 3]"},
                {"input": "sort_and_deduplicate([])", "expected": "[]"},
                {"input": "sort_and_deduplicate([1])", "expected": "[1]"},
                {"input": "sort_and_deduplicate([5,5,5])", "expected": "[5]"},
            ]

        request = VerificationRequest(
            code=code,
            language="python",
            test_suite=test_suite,
            acp_job_id=job["id"],
            provider_address=self.agent_address,
            buyer_address=job["buyer"],
            service_description=job["requirement"],
        )

        result = self.verifier.verify(request)

        if result.verified:
            print(f"   ✅ Verified! {result.passed}/{result.total} tests passed ({result.verification_time_ms}ms)")
        elif result.error:
            print(f"   ⚠️  Verification unavailable: {result.error}")
            print(f"   → Delivering without verification (fallback)")
        else:
            print(f"   ❌ Failed! {result.passed}/{result.total} tests passed")
            print(f"   → Should fix code before delivering!")

        return result

    def _deliver(self, job: dict, code: str, verification: VerificationResult):
        """Deliver work to buyer via ACP."""
        memo = format_acp_delivery_memo(
            deliverable=code,
            verification=verification,
            notes=f"Delivered by {self.agent_name}" + (
                " — ClawBizarre pre-verified ✓" if verification.verified
                else " — verification unavailable" if verification.error
                else " — ⚠️ some tests failed"
            )
        )

        print(f"\n📤 Delivering to buyer...")
        print(f"   Verification status: {'✅ verified' if verification.verified else '⚠️ unverified'}")

        # In production: acp.deliver_job(job['id'], json.dumps(memo))
        print(f"\n   ACP Memo (verification section):")
        print(f"   {json.dumps(memo['verification'], indent=4)}")


def main():
    print("=" * 60)
    print("ACP Provider Agent — ClawBizarre Pre-Verification Example")
    print("=" * 60)

    provider = MockACPProvider()

    # Simulate receiving a job from ACP
    mock_job = {
        "id": "acp-onchain-42",
        "buyer": "0xBuyerAgent...",
        "requirement": "Write a Python function that sorts a list and removes duplicates",
        "price": "0.01",  # USDC
        "test_suite": [
            {"input": "sort_and_deduplicate([3,1,2,1,3])", "expected": "[1, 2, 3]"},
            {"input": "sort_and_deduplicate([])", "expected": "[]"},
            {"input": "sort_and_deduplicate([1])", "expected": "[1]"},
            {"input": "sort_and_deduplicate([5,5,5])", "expected": "[5]"},
        ]
    }

    provider.on_new_task(mock_job)

    # Show what buyer sees
    print("\n" + "=" * 60)
    print("BUYER'S PERSPECTIVE")
    print("=" * 60)
    print("""
When buyer receives the delivery memo, they see:
  - The deliverable (code)
  - ClawBizarre verification: ✅ 4/4 tests passed
  - VRF receipt hash for independent verification
  
This is STRICTLY better than unverified delivery:
  - Buyer can trust the code works (structural proof)
  - Buyer can skip their own evaluation (saves compute)
  - If buyer has their own evaluator, the receipt is confirmatory evidence

Provider benefits:
  - Higher trust → more likely to get paid
  - Builds reputation for verified deliveries
  - Differentiates from unverified providers
  - Cost: $0.005/verification (negligible vs job price)
""")


if __name__ == "__main__":
    main()
