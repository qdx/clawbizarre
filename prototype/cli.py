#!/usr/bin/env python3
"""
clawbizarre CLI — manage agent identity, sign/verify work receipts, inspect chains.

Usage:
  python3 cli.py init [--keydir DIR]          Generate a new agent identity
  python3 cli.py whoami [--keydir DIR]        Show current identity
  python3 cli.py receipt create [options]     Create and sign a work receipt
  python3 cli.py receipt verify FILE          Verify a signed receipt
  python3 cli.py chain append FILE RECEIPT    Append receipt to chain
  python3 cli.py chain verify FILE            Verify chain integrity
  python3 cli.py chain stats FILE             Show chain statistics
"""

import argparse
import json
import os
import sys
from pathlib import Path

from identity import AgentIdentity, SignedReceipt, sign_receipt
from receipt import (
    WorkReceipt, TestResults, VerificationTier, ReceiptChain,
    hash_content,
)

DEFAULT_KEYDIR = os.path.expanduser("~/.clawbizarre")


def cmd_init(args):
    """Generate a new agent identity."""
    keydir = Path(args.keydir)
    keyfile = keydir / "agent.pem"

    if keyfile.exists() and not args.force:
        print(f"Identity already exists at {keyfile}. Use --force to overwrite.")
        sys.exit(1)

    identity = AgentIdentity.generate()
    identity.save_keyfile(str(keyfile))

    # Save public info
    info = {
        "agent_id": identity.agent_id,
        "fingerprint": identity.fingerprint,
        "public_key": identity.public_key_hex,
    }
    (keydir / "identity.json").write_text(json.dumps(info, indent=2))

    print(f"Identity generated:")
    print(f"  Agent ID:    {identity.agent_id}")
    print(f"  Fingerprint: {identity.fingerprint}")
    print(f"  Key saved:   {keyfile}")
    print(f"  Public info: {keydir / 'identity.json'}")


def cmd_whoami(args):
    """Show current identity."""
    keydir = Path(args.keydir)
    info_file = keydir / "identity.json"
    if not info_file.exists():
        print("No identity found. Run `clawbizarre init` first.")
        sys.exit(1)
    info = json.loads(info_file.read_text())
    print(f"Agent ID:    {info['agent_id']}")
    print(f"Fingerprint: {info['fingerprint']}")
    print(f"Public key:  {info['public_key']}")


def _load_identity(keydir: str) -> AgentIdentity:
    keyfile = Path(keydir) / "agent.pem"
    if not keyfile.exists():
        print("No identity found. Run `clawbizarre init` first.")
        sys.exit(1)
    return AgentIdentity.from_keyfile(str(keyfile))


def cmd_receipt_create(args):
    """Create and sign a work receipt."""
    identity = _load_identity(args.keydir)

    # Build test results if provided
    test_results = None
    if args.tests_passed is not None:
        test_results = TestResults(
            passed=args.tests_passed,
            failed=args.tests_failed or 0,
            suite_hash=args.suite_hash or hash_content("unknown"),
        )

    receipt = WorkReceipt(
        agent_id=identity.agent_id,
        task_type=args.task_type,
        verification_tier=VerificationTier(args.tier),
        input_hash=args.input_hash or hash_content(args.input or ""),
        output_hash=args.output_hash or hash_content(args.output or ""),
        platform=args.platform or "cli",
        test_results=test_results,
        environment_hash=args.env_hash,
        agent_config_hash=args.config_hash,
    )

    signed = sign_receipt(identity, receipt)

    if args.output_file:
        Path(args.output_file).write_text(signed.to_json())
        print(f"Signed receipt written to {args.output_file}")
    else:
        print(signed.to_json())

    print(f"\nReceipt ID:   {receipt.receipt_id}")
    print(f"Content hash: {receipt.content_hash}")
    print(f"Tier:         {receipt.verification_tier.name}")
    if test_results:
        print(f"Tests:        {test_results.passed} passed, {test_results.failed} failed → {'✓' if test_results.success else '✗'}")


def cmd_receipt_verify(args):
    """Verify a signed receipt."""
    data = Path(args.file).read_text()
    signed = SignedReceipt.from_json(data)

    # Extract public key from signer_id
    if signed.signer_id.startswith("ed25519:"):
        pubkey_hex = signed.signer_id.split(":", 1)[1]
        verifier = AgentIdentity.from_public_key_hex(pubkey_hex)
        valid = signed.verify(verifier)
    else:
        print(f"Unknown signer format: {signed.signer_id}")
        sys.exit(1)

    # Also verify content hash matches receipt
    from receipt import WorkReceipt as WR
    receipt = WR.from_json(signed.receipt_json)
    hash_match = receipt.content_hash == signed.content_hash

    print(f"Signer:       {signed.signer_id[:50]}...")
    print(f"Content hash: {signed.content_hash}")
    print(f"Signature:    {'✓ VALID' if valid else '✗ INVALID'}")
    print(f"Hash match:   {'✓' if hash_match else '✗ MISMATCH'}")

    if valid and hash_match:
        print("\n✓ Receipt is authentic and unmodified")
        # Show receipt details
        print(f"  Task:    {receipt.task_type}")
        print(f"  Tier:    {receipt.verification_tier.name}")
        print(f"  Time:    {receipt.timestamp}")
        if receipt.test_results:
            print(f"  Tests:   {receipt.test_results.passed}p/{receipt.test_results.failed}f")
    else:
        print("\n✗ Receipt FAILED verification")
        sys.exit(1)


def cmd_chain_append(args):
    """Append a signed receipt to a chain file."""
    chain_path = Path(args.chain_file)

    # Load or create chain
    if chain_path.exists():
        chain_data = json.loads(chain_path.read_text())
        chain = ReceiptChain()
        for entry in chain_data.get("entries", []):
            receipt = WorkReceipt.from_json(json.dumps(entry["receipt"]))
            chain.append(receipt)
    else:
        chain = ReceiptChain()

    # Load receipt
    signed_data = Path(args.receipt_file).read_text()
    signed = SignedReceipt.from_json(signed_data)
    receipt = WorkReceipt.from_json(signed.receipt_json)

    chain.append(receipt)

    # Save chain
    entries = []
    for r, h in zip(chain.receipts, chain.chain_hashes):
        entries.append({
            "receipt": json.loads(r.to_json()),
            "chain_hash": h,
        })
    chain_path.write_text(json.dumps({"entries": entries}, indent=2))
    print(f"Appended receipt to chain ({chain.length} entries)")
    print(f"Chain integrity: {'✓' if chain.verify_integrity() else '✗'}")


def cmd_chain_verify(args):
    """Verify chain integrity."""
    chain_data = json.loads(Path(args.chain_file).read_text())
    chain = ReceiptChain()
    for entry in chain_data.get("entries", []):
        receipt = WorkReceipt.from_json(json.dumps(entry["receipt"]))
        chain.append(receipt)

    # Compare computed hashes with stored hashes
    stored_hashes = [e["chain_hash"] for e in chain_data["entries"]]
    match = all(a == b for a, b in zip(chain.chain_hashes, stored_hashes))

    print(f"Chain length:    {chain.length}")
    print(f"Hash integrity:  {'✓' if chain.verify_integrity() else '✗'}")
    print(f"Stored matches:  {'✓' if match else '✗ TAMPERED'}")

    if not match:
        for i, (a, b) in enumerate(zip(chain.chain_hashes, stored_hashes)):
            if a != b:
                print(f"  Mismatch at entry {i}: expected {b[:20]}..., got {a[:20]}...")


def cmd_chain_stats(args):
    """Show chain statistics."""
    chain_data = json.loads(Path(args.chain_file).read_text())
    chain = ReceiptChain()
    agents = set()
    task_types = {}
    for entry in chain_data.get("entries", []):
        receipt = WorkReceipt.from_json(json.dumps(entry["receipt"]))
        chain.append(receipt)
        agents.add(receipt.agent_id[:30])
        task_types[receipt.task_type] = task_types.get(receipt.task_type, 0) + 1

    print(f"Chain length:   {chain.length}")
    print(f"Integrity:      {'✓' if chain.verify_integrity() else '✗'}")
    print(f"Unique agents:  {len(agents)}")
    print(f"Tier breakdown: {chain.tier_breakdown()}")
    print(f"Tier 0 success: {chain.success_rate():.0%}")
    print(f"Task types:")
    for t, c in sorted(task_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


def main():
    parser = argparse.ArgumentParser(prog="clawbizarre", description="Agent identity & work receipt CLI")
    parser.add_argument("--keydir", default=DEFAULT_KEYDIR, help="Directory for identity keys")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Generate new agent identity")
    p_init.add_argument("--force", action="store_true")

    # whoami
    sub.add_parser("whoami", help="Show current identity")

    # receipt
    p_receipt = sub.add_parser("receipt", help="Work receipt operations")
    receipt_sub = p_receipt.add_subparsers(dest="receipt_cmd")

    p_create = receipt_sub.add_parser("create", help="Create and sign a receipt")
    p_create.add_argument("--task-type", required=True)
    p_create.add_argument("--tier", type=int, default=0, choices=[0, 1, 2, 3])
    p_create.add_argument("--input", default="")
    p_create.add_argument("--output", default="")
    p_create.add_argument("--input-hash")
    p_create.add_argument("--output-hash")
    p_create.add_argument("--platform", default="cli")
    p_create.add_argument("--tests-passed", type=int)
    p_create.add_argument("--tests-failed", type=int, default=0)
    p_create.add_argument("--suite-hash")
    p_create.add_argument("--env-hash")
    p_create.add_argument("--config-hash")
    p_create.add_argument("-o", "--output-file")

    p_verify = receipt_sub.add_parser("verify", help="Verify a signed receipt")
    p_verify.add_argument("file")

    # chain
    p_chain = sub.add_parser("chain", help="Receipt chain operations")
    chain_sub = p_chain.add_subparsers(dest="chain_cmd")

    p_append = chain_sub.add_parser("append", help="Append receipt to chain")
    p_append.add_argument("chain_file")
    p_append.add_argument("receipt_file")

    p_cverify = chain_sub.add_parser("verify", help="Verify chain integrity")
    p_cverify.add_argument("chain_file")

    p_stats = chain_sub.add_parser("stats", help="Chain statistics")
    p_stats.add_argument("chain_file")

    args, unknown = parser.parse_known_args()

    # Propagate keydir to all subcommands
    if not hasattr(args, 'keydir') or args.keydir is None:
        args.keydir = DEFAULT_KEYDIR

    if args.command == "init":
        cmd_init(args)
    elif args.command == "whoami":
        cmd_whoami(args)
    elif args.command == "receipt":
        if args.receipt_cmd == "create":
            cmd_receipt_create(args)
        elif args.receipt_cmd == "verify":
            cmd_receipt_verify(args)
        else:
            p_receipt.print_help()
    elif args.command == "chain":
        if args.chain_cmd == "append":
            cmd_chain_append(args)
        elif args.chain_cmd == "verify":
            cmd_chain_verify(args)
        elif args.chain_cmd == "stats":
            cmd_chain_stats(args)
        else:
            p_chain.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
