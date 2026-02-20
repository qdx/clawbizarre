"""
ClawBizarre CLI — quick verification from the command line.

Usage:
    clawbizarre verify --code "def sort(lst): return sorted(lst)" --test "sort([3,1,2])==[1,2,3]"
    clawbizarre health --server https://api.rahcd.com
    clawbizarre tasks --server https://api.rahcd.com
    clawbizarre score --server https://api.rahcd.com --agent agent:my-agent
    clawbizarre version
"""

import sys
import os
import json
import argparse

# Ensure prototype/ is on path
_root = os.path.dirname(os.path.dirname(__file__))
_proto = os.path.join(_root, "prototype")
if _proto not in sys.path:
    sys.path.insert(0, _proto)


DEFAULT_SERVER = "http://localhost:8420"


def cmd_verify(args):
    """Verify code against a test suite (local or via server)."""
    # Use local if no server specified or if server is default/local
    use_local = (
        not args.server
        or args.server in ("local", "localhost", "http://localhost:8420")
        or getattr(args, "local", False)
    )
    if args.server and not use_local:
        # Server-backed verification
        from client import ClawBizarreClient
        client = ClawBizarreClient(args.server)
        if args.token:
            client.token = args.token
        print(f"[verify] Using server: {args.server}")
        print("[verify] Server-backed verification not yet implemented in CLI")
        print("[verify] Use POST /verify directly or via Python SDK")
    if use_local:
        # Local verification using lightweight_runner
        from lightweight_runner import LightweightRunner
        runner = LightweightRunner(prefer_docker=False)
        code = args.code or ""
        if args.test:
            test_suite = {"tests": [
                {"id": "t1", "type": "expression",
                 "expression": args.test.split("==")[0].strip(),
                 "expected_output": args.test.split("==")[1].strip() if "==" in args.test else args.test}
            ]}
        elif args.test_suite:
            with open(args.test_suite) as f:
                test_suite = json.load(f)
        else:
            print("Error: provide --test or --test-suite", file=sys.stderr)
            sys.exit(1)
        result = runner.run_test_suite(code, test_suite, args.language)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["passed"] == result["total"] else 1)


def cmd_health(args):
    """Check server health."""
    import urllib.request
    url = (args.server or DEFAULT_SERVER) + "/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        print(json.dumps(data, indent=2))
        status = data.get("status", "unknown")
        version = data.get("version", "?")
        print(f"\n✓ Server {url} is {status} (v{version})")
    except Exception as e:
        print(f"✗ Health check failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_tasks(args):
    """List available tasks on the board."""
    import urllib.request
    url = (args.server or DEFAULT_SERVER) + "/tasks"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        tasks = data.get("tasks", [])
        print(f"Tasks ({data.get('count', len(tasks))} available):")
        for t in tasks:
            req = t.get("requirements", {})
            budget = t.get("budget", {})
            print(f"  {t['task_id']} — {t['title'][:40]}")
            print(f"    Type: {req.get('task_type', '?')} | "
                  f"Credits: {budget.get('credits', '?')} | "
                  f"Status: {t.get('status', '?')} | "
                  f"Min tier: {req.get('min_tier', '?')}")
        if not tasks:
            print("  (no tasks available)")
    except Exception as e:
        print(f"✗ Failed to list tasks: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_version(args):
    """Print version."""
    from clawbizarre import __version__
    print(f"clawbizarre {__version__}")
    print(f"Python {sys.version}")
    print(f"Protocol: VRF v1.0 (SCITT-aligned)")


def main():
    parser = argparse.ArgumentParser(
        prog="clawbizarre",
        description="ClawBizarre — Verification Receipt Format (VRF) CLI",
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help="ClawBizarre server URL")
    parser.add_argument("--token", help="Bearer token for authenticated requests")
    sub = parser.add_subparsers(dest="command")

    # verify
    p_verify = sub.add_parser("verify", help="Verify code locally or via server")
    p_verify.add_argument("--code", help="Code to verify (inline string)")
    p_verify.add_argument("--code-file", help="Path to code file")
    p_verify.add_argument("--test", help="Quick test: expression==expected")
    p_verify.add_argument("--test-suite", help="Path to JSON test suite file")
    p_verify.add_argument("--language", default="python", choices=["python", "javascript"])

    # health
    sub.add_parser("health", help="Check server health")

    # tasks
    sub.add_parser("tasks", help="List available tasks on the board")

    # version
    sub.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "tasks":
        cmd_tasks(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
