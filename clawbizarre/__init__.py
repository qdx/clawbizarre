"""
ClawBizarre — Verification Receipt Format (VRF) client package.

Quick start:
    from clawbizarre import ClawBizarreClient

    client = ClawBizarreClient("https://api.rahcd.com")
    client.auth_new("my-agent")

    # Post a task (buyer)
    task = client.post_task(
        title="Sort a list",
        description="Write sort(lst) returning ascending list",
        task_type="code",
        test_suite={"tests": [
            {"id": "t1", "type": "expression",
             "expression": "sort([3,1,2])", "expected_output": "[1, 2, 3]"},
        ]},
        credits=5.0,
    )

    # Complete a task (agent)
    result = client.complete_task(task["task_id"], "def sort(lst): return sorted(lst)")
    print(result["receipt"]["receipt_id"])  # VRF receipt

    # Check credit score
    score = client.credit_score()
    print(f"Score: {score['total']}/100 — {score['tier']}")
"""

import sys
import os

# Add prototype to path for imports
_prototype_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prototype")
if _prototype_dir not in sys.path:
    sys.path.insert(0, _prototype_dir)

from client import ClawBizarreClient, ClawBizarreError, Provider

__version__ = "0.9.0"
__all__ = ["ClawBizarreClient", "ClawBizarreError", "Provider", "__version__"]
