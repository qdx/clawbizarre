"""
ClawBizarre Evaluator Benchmark
Tests verify_server against progressively harder real-world coding tasks.
Validates that structural verification catches real bugs.

Usage:
    python3 benchmark_evaluator.py [--test]
"""

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

# ── Benchmark Tasks ──────────────────────────────────────────────

TASKS = [
    # Level 1: Pure functions
    {
        "name": "fibonacci",
        "description": "Implement fibonacci(n) returning nth Fibonacci number",
        "good_code": "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b",
        "bad_code": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)  # correct but will timeout on large n",
        "buggy_code": "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return a  # off-by-one bug",
        "tests": [
            {"name": "fib_0", "input": "fibonacci(0)", "expected_output": "0"},
            {"name": "fib_1", "input": "fibonacci(1)", "expected_output": "1"},
            {"name": "fib_10", "input": "fibonacci(10)", "expected_output": "55"},
            {"name": "fib_20", "input": "fibonacci(20)", "expected_output": "6765"},
        ]
    },
    # Level 2: Data structures
    {
        "name": "lru_cache",
        "description": "Implement LRUCache with get(key) and put(key, value)",
        "good_code": """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
""",
        "buggy_code": """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
    def get(self, key):
        return self.cache.get(key, -1)  # doesn't update access order
    def put(self, key, value):
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.pop(next(iter(self.cache)))  # evicts first inserted, not LRU
""",
        "tests": [
            {"name": "put_get", "input": "(lambda c: (c.put(1, 'a'), c.get(1))[1])(LRUCache(2))", "expected_output": "'a'"},
            {"name": "eviction", "input": "(lambda c: (c.put(1,'a'), c.put(2,'b'), c.put(3,'c'), c.get(1))[3])(LRUCache(2))", "expected_output": "-1"},
            {"name": "access_refresh", "input": "(lambda c: (c.put(1,'a'), c.put(2,'b'), c.get(1), c.put(3,'c'), c.get(2))[4])(LRUCache(2))", "expected_output": "-1"},
            {"name": "overwrite", "input": "(lambda c: (c.put(1,'a'), c.put(1,'b'), c.get(1))[2])(LRUCache(2))", "expected_output": "'b'"},
        ]
    },
    # Level 3: String algorithms
    {
        "name": "anagram_groups",
        "description": "Group anagrams from a list of strings",
        "good_code": """
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = ''.join(sorted(s))
        groups.setdefault(key, []).append(s)
    return sorted([sorted(g) for g in groups.values()])
""",
        "buggy_code": """
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = ''.join(sorted(s.lower()))  # bug: lowercases when shouldn't
        groups.setdefault(key, []).append(s)
    return sorted([sorted(g) for g in groups.values()])
""",
        "tests": [
            {"name": "basic", "input": "group_anagrams(['eat','tea','tan','ate','nat','bat'])", "expected_output": "[['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]"},
            {"name": "empty", "input": "group_anagrams([''])", "expected_output": "[['']]"},
            {"name": "case_sensitive", "input": "group_anagrams(['Eat','tea','Tea'])", "expected_output": "[['Eat'], ['Tea'], ['tea']]"},
        ]
    },
    # Level 4: Binary search variant
    {
        "name": "search_rotated",
        "description": "Search in rotated sorted array",
        "good_code": """
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
""",
        "buggy_code": """
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] < nums[mid]:  # bug: should be <= (fails when lo==mid)
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
""",
        "tests": [
            {"name": "found", "input": "search_rotated([4,5,6,7,0,1,2], 0)", "expected_output": "4"},
            {"name": "not_found", "input": "search_rotated([4,5,6,7,0,1,2], 3)", "expected_output": "-1"},
            {"name": "two_elem", "input": "search_rotated([3,1], 1)", "expected_output": "1"},
            {"name": "single", "input": "search_rotated([1], 1)", "expected_output": "0"},
        ]
    },
    # Level 5: Complex class with state
    {
        "name": "task_scheduler",
        "description": "Priority task scheduler with dependencies",
        "good_code": """
from collections import defaultdict, deque
class TaskScheduler:
    def __init__(self):
        self.tasks = {}
        self.deps = defaultdict(set)
        self.rdeps = defaultdict(set)
    def add_task(self, name, priority=0, depends_on=None):
        self.tasks[name] = priority
        for d in (depends_on or []):
            self.deps[name].add(d)
            self.rdeps[d].add(name)
    def get_order(self):
        in_degree = {t: len(self.deps[t]) for t in self.tasks}
        ready = sorted([t for t in self.tasks if in_degree[t] == 0],
                       key=lambda t: -self.tasks[t])
        order = []
        while ready:
            t = ready.pop(0)
            order.append(t)
            for dep in self.rdeps[t]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    ready.append(dep)
                    ready.sort(key=lambda x: -self.tasks[x])
        return order if len(order) == len(self.tasks) else None  # None = cycle
""",
        "buggy_code": """
class TaskScheduler:
    def __init__(self):
        self.tasks = {}
        self.deps = {}
    def add_task(self, name, priority=0, depends_on=None):
        self.tasks[name] = priority
        self.deps[name] = set(depends_on or [])
    def get_order(self):
        return sorted(self.tasks.keys(), key=lambda t: -self.tasks[t])  # ignores deps entirely
""",
        "tests": [
            {"name": "no_deps", "input": "(lambda s: (s.add_task('a', 1), s.add_task('b', 2), s.get_order())[2])(TaskScheduler())", "expected_output": "['b', 'a']"},
            {"name": "with_deps", "input": "(lambda s: (s.add_task('a', 1), s.add_task('b', 2, ['a']), s.get_order())[2])(TaskScheduler())", "expected_output": "['a', 'b']"},
            {"name": "chain", "input": "(lambda s: (s.add_task('c', 3), s.add_task('b', 2, ['c']), s.add_task('a', 1, ['b']), s.get_order())[3])(TaskScheduler())", "expected_output": "['c', 'b', 'a']"},
        ]
    },
]


@dataclass
class BenchmarkResult:
    task_name: str
    code_type: str  # good, bad, buggy
    verdict: str
    passed: int
    total: int
    time_ms: float
    correct_detection: bool  # did verification correctly identify good/bad?


def run_benchmark(base_url: str = "http://127.0.0.1:8700") -> list:
    results = []
    
    for task in TASKS:
        for code_type in ["good_code", "buggy_code"]:
            if code_type not in task:
                continue
                
            payload = {
                "output": {"content": task[code_type]},
                "verification": {
                    "tier": 0,
                    "test_suite": {"tests": task["tests"]}
                }
            }
            
            start = time.monotonic()
            try:
                req = urllib.request.Request(
                    f"{base_url}/verify",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
                elapsed = (time.monotonic() - start) * 1000
                
                verdict = resp["verdict"]
                passed = resp["results"]["passed"]
                total = resp["results"]["total"]
                
                # Good code should pass, buggy code should fail
                expected_pass = code_type == "good_code"
                actual_pass = verdict == "pass"
                correct = expected_pass == actual_pass
                
                results.append(BenchmarkResult(
                    task_name=task["name"],
                    code_type=code_type.replace("_code", ""),
                    verdict=verdict,
                    passed=passed,
                    total=total,
                    time_ms=elapsed,
                    correct_detection=correct
                ))
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                results.append(BenchmarkResult(
                    task_name=task["name"],
                    code_type=code_type.replace("_code", ""),
                    verdict=f"error: {e}",
                    passed=0,
                    total=len(task["tests"]),
                    time_ms=elapsed,
                    correct_detection=False
                ))
    
    return results


def print_results(results: list):
    print("\n" + "=" * 70)
    print("ClawBizarre Evaluator Benchmark Results")
    print("=" * 70)
    
    correct = sum(1 for r in results if r.correct_detection)
    total = len(results)
    
    for r in results:
        icon = "✅" if r.correct_detection else "❌"
        print(f"  {icon} {r.task_name:20s} [{r.code_type:5s}] → {r.verdict:6s} "
              f"({r.passed}/{r.total}) {r.time_ms:.0f}ms")
    
    print(f"\nDetection accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    avg_ms = sum(r.time_ms for r in results) / len(results)
    print(f"Average verification time: {avg_ms:.0f}ms")
    print(f"Total time: {sum(r.time_ms for r in results):.0f}ms")


def run_tests():
    """Self-test: start server, run benchmark, verify accuracy."""
    import http.server
    from threading import Thread
    from verify_server import VerifyHandler, VerificationEngine
    
    try:
        from identity import AgentIdentity
        identity = AgentIdentity.generate()
    except ImportError:
        identity = None
    
    engine = VerificationEngine(identity=identity)
    VerifyHandler.engine = engine
    server = http.server.HTTPServer(("127.0.0.1", 0), VerifyHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    
    results = run_benchmark(f"http://127.0.0.1:{port}")
    print_results(results)
    
    server.shutdown()
    
    correct = sum(1 for r in results if r.correct_detection)
    total = len(results)
    
    print(f"\n{'PASS' if correct == total else 'FAIL'}: {correct}/{total} correct detections")
    return correct == total


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        results = run_benchmark()
        print_results(results)
