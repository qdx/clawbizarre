"""
ClawBizarre End-to-End Integration Test
Wires together: Identity → Signed Handshake → Receipts → Aggregator → Discovery → Treasury

Full lifecycle: Alice (provider) and Bob (requester) negotiate, execute, verify,
build reputation, discover each other, and process payment.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta

from identity import AgentIdentity
from receipt import (
    WorkReceipt, ReceiptChain, TestResults, Timing,
    VerificationTier, PricingStrategy, hash_content,
)
from signed_handshake import SignedHandshakeSession
from reputation import MerkleTree
from aggregator import ReputationAggregator, ReputationSnapshot
from discovery import CapabilityAd, Registry
from treasury import BudgetPolicy, TreasuryAgent


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run():
    now = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ==== Phase 1: Identity ====
        section("Phase 1: Identity Generation")

        alice_id = AgentIdentity.generate()
        alice_id.save_keyfile(os.path.join(tmpdir, "alice.key"))
        bob_id = AgentIdentity.generate()
        bob_id.save_keyfile(os.path.join(tmpdir, "bob.key"))
        print(f"Alice: {alice_id.agent_id[:40]}...")
        print(f"Bob:   {bob_id.agent_id[:40]}...")

        alice_restored = AgentIdentity.from_keyfile(os.path.join(tmpdir, "alice.key"))
        assert alice_restored.agent_id == alice_id.agent_id
        print("✓ Identity persistence verified")

        # ==== Phase 2: Signed Handshake ====
        section("Phase 2: Signed Handshake")

        alice_hs = SignedHandshakeSession(alice_id)
        bob_hs = SignedHandshakeSession(bob_id)

        # Exchange hellos
        a_hello = alice_hs.send_hello(["code_review", "research"])
        assert bob_hs.receive_hello(a_hello)
        b_hello = bob_hs.send_hello(["data_validation"])
        assert alice_hs.receive_hello(b_hello)
        print("Hellos exchanged ✓")

        # Alice proposes code review task
        test_suite_src = "assert issues_found >= 0"
        proposal = alice_hs.propose(
            task_description="Review PR #42 for security issues",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            test_suite_hash=hash_content(test_suite_src),
            input_data="def transfer(amt): db.execute(f'UPDATE bal SET v={amt}')",
        )
        assert bob_hs.receive_proposal(proposal)
        print("Proposal accepted ✓")

        # Bob accepts and executes
        accept_msg = bob_hs.accept()
        assert alice_hs.receive_accept(accept_msg)

        execute_msg = bob_hs.execute(
            output="SQL injection found — use parameterized queries",
            proof={"issues_found": 1},
        )
        print("Work executed ✓")

        # Alice verifies & generates signed receipt
        def verifier(payload):
            output = payload.get("output", "")
            ok = "sql injection" in output.lower()
            return TestResults(
                passed=1 if ok else 0,
                failed=0 if ok else 1,
                suite_hash=hash_content(test_suite_src),
            )

        verify_msg, signed_receipt = alice_hs.verify_and_receipt(execute_msg, verifier)
        assert signed_receipt is not None, "Should produce signed receipt"
        assert signed_receipt.verify(alice_id), "Receipt signature must verify"
        print(f"Signed receipt: {signed_receipt.content_hash[:40]}...")
        print("✓ All handshake signatures verified")

        # Extract the underlying WorkReceipt
        receipt = alice_hs.session.receipt
        assert receipt.verify_tier0(), "Receipt should be Tier 0 verified"

        # ==== Phase 3: Receipt Chain & Aggregation ====
        section("Phase 3: Receipt Chain + Reputation Aggregation")

        alice_chain = ReceiptChain()

        # Historical receipts (25 past tasks)
        for i in range(25):
            ts = (now - timedelta(days=50 - i * 2)).isoformat()
            hist = WorkReceipt(
                agent_id=alice_id.agent_id,
                task_type="code_review" if i % 3 != 0 else "research",
                verification_tier=VerificationTier.SELF_VERIFYING,
                input_hash=hash_content(f"hist_in_{i}"),
                output_hash=hash_content(f"hist_out_{i}"),
                timestamp=ts,
                pricing_strategy=PricingStrategy.REPUTATION_PREMIUM,
                test_results=TestResults(
                    passed=5 if i != 12 else 3,
                    failed=0 if i != 12 else 1,
                    suite_hash=hash_content("suite"),
                ),
                timing=Timing(
                    started_at=ts,
                    completed_at=ts,
                    deadline=(now - timedelta(days=50 - i * 2) + timedelta(hours=4)).isoformat(),
                ),
            )
            alice_chain.append(hist)

        # Add the live receipt
        alice_chain.append(receipt)

        print(f"Chain length: {alice_chain.length}")
        assert alice_chain.verify_integrity()
        print("Chain integrity: ✓")

        agg = ReputationAggregator()
        alice_snap = agg.aggregate(alice_chain)

        print(f"Composite score:      {alice_snap.composite_score}")
        print(f"Success rate:         {alice_snap.success_rate}")
        print(f"Strategy consistency: {alice_snap.strategy_consistency}")
        print(f"On-time rate:         {alice_snap.on_time_rate}")
        print(f"Trust tier:           {alice_snap.trust_tier}")
        print(f"Domain scores:        {alice_snap.domain_scores}")
        print(f"Merkle root:          {alice_snap.merkle_root[:16]}...")

        assert alice_snap.composite_score > 0.7
        assert alice_snap.trust_tier == "established"  # 26 receipts, 20-50 range

        # Portability round-trip
        restored = ReputationSnapshot.from_json(alice_snap.to_json())
        assert restored.content_hash == alice_snap.content_hash
        print("✓ Snapshot portable and verified")

        # ==== Phase 4: Discovery ====
        section("Phase 4: Discovery")

        registry = Registry()

        alice_ad = CapabilityAd(
            agent_id=alice_id.agent_id,
            capabilities=["code_review", "research"],
            verification_tier=0,
            pricing_strategy="reputation_premium",
            receipt_chain_length=alice_snap.chain_length,
            success_rate=alice_snap.success_rate,
            on_time_rate=alice_snap.on_time_rate or 0.0,
            strategy_consistency=alice_snap.strategy_consistency,
            endpoint="https://alice.agent.local/api",
        )
        registry.register(alice_ad)

        # Register competitors
        for name, score, length in [
            ("charlie", 0.85, 30),
            ("diana", 0.70, 5),
            ("eve", 0.92, 55),
        ]:
            registry.register(CapabilityAd(
                agent_id=f"sigil:{name}",
                capabilities=["code_review"],
                verification_tier=0,
                pricing_strategy="reputation_premium",
                receipt_chain_length=length,
                success_rate=score,
                on_time_rate=0.9,
                strategy_consistency=0.95,
                endpoint=f"https://{name}.agent.local/api",
            ))

        from discovery import SearchQuery
        results = registry.search(SearchQuery(task_type="code_review"))
        print(f"Search results: {len(results)} agents")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r.agent_id[:30]:30s} score={r.relevance_score:.3f} len={r.capability_ad.receipt_chain_length}")

        alice_found = any(r.agent_id == alice_id.agent_id for r in results)
        assert alice_found, "Alice not found in discovery"
        print("✓ Alice discovered")

        # ==== Phase 5: Treasury ====
        section("Phase 5: Treasury Agent")

        from treasury import SpendRequest, SpendCategory, ApprovalDecision
        policy = BudgetPolicy(
            daily_budget=50.0,
            weekly_budget=200.0,
            monthly_budget=500.0,
            auto_approve_threshold=10.0,
            escalation_threshold=50.0,
        )
        treasury = TreasuryAgent(policy)

        req1 = SpendRequest(
            requesting_agent=bob_id.agent_id,
            counterparty=alice_id.agent_id,
            amount=5.0,
            category=SpendCategory.SERVICE,
            description="Code review of PR #42",
        )
        decision = treasury.evaluate(req1)
        print(f"Decision: {decision.decision.value} — {decision.reason}")
        assert decision.decision == ApprovalDecision.AUTO_APPROVE

        # Test a blocked counterparty
        policy.blocked_counterparties.append("sigil:evil_agent")
        req2 = SpendRequest(
            counterparty="sigil:evil_agent", amount=1.0,
            category=SpendCategory.SERVICE, description="test",
        )
        blocked = treasury.evaluate(req2)
        assert blocked.decision == ApprovalDecision.REJECTED
        print(f"Blocked:  {blocked.decision.value} — {blocked.reason}")

        # Test escalation
        req3 = SpendRequest(
            counterparty="sigil:charlie", amount=45.0,
            category=SpendCategory.SERVICE, description="big job",
        )
        big = treasury.evaluate(req3)
        print(f"Big job:  {big.decision.value} — {big.reason}")

        audit = treasury.audit_log
        print(f"Audit entries: {len(audit)}")
        print("✓ Treasury operational")

        # ==== Phase 6: Full Chain of Trust ====
        section("Phase 6: Chain of Trust Verification")

        checks = [
            ("Identity → Handshake", signed_receipt.verify(alice_id)),
            ("Handshake → Receipt", receipt.agent_id == bob_id.agent_id),  # Bob did the work
            ("Receipt → Chain", alice_chain.verify_integrity()),
            ("Chain → Snapshot", alice_snap.chain_length == alice_chain.length),
            ("Snapshot → Merkle", MerkleTree(
                [r.content_hash.replace("sha256:", "") for r in alice_chain.receipts]
            ).root == alice_snap.merkle_root),
            ("Snapshot → Discovery", alice_ad.receipt_chain_length == alice_snap.chain_length),
            ("Discovery → Found", alice_found),
            ("Found → Treasury", decision.decision == ApprovalDecision.AUTO_APPROVE),
        ]

        all_ok = True
        for label, ok in checks:
            status = "✓" if ok else "✗"
            print(f"  {status} {label}")
            if not ok:
                all_ok = False

        print()
        if all_ok:
            print("🎉 END-TO-END INTEGRATION: ALL PHASES PASSED")
        else:
            print("❌ SOME CHECKS FAILED")
            raise AssertionError("Integration test failed")

        print(f"\nSummary:")
        print(f"  Identities:    2")
        print(f"  Handshake:     6 signed messages")
        print(f"  Receipt chain: {alice_chain.length} receipts")
        print(f"  Reputation:    {alice_snap.composite_score} ({alice_snap.trust_tier})")
        print(f"  Discovery:     {len(results)} results")
        print(f"  Treasury:      {len(treasury.audit_log)} decisions")


if __name__ == "__main__":
    run()
