"""
ClawBizarre Discovery MVP — Phase 2
Simple capability registry with newcomer protection (Law 6).

Design decisions:
- Pull-only for Era 1 (no push notifications yet)
- JSON registry served via simple HTTP
- 30% discovery slots reserved for newcomers (<20 receipts)
- Search by task_type, verification_tier, pricing_strategy
- Ranked by reputation score with newcomer boost

Based on simulation Law 6: NewcomerHub (30% reserved) consistently became
the LARGEST marketplace across simulations (49 agents, 3500-5700 tasks).
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class AvailabilityStatus(str, Enum):
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class CapabilityAd:
    """What an agent advertises to the registry."""
    agent_id: str  # ed25519 pubkey
    capabilities: list[str]  # ["code_review", "translation_en_zh", ...]
    verification_tier: int  # max tier this agent supports
    availability: AvailabilityStatus = AvailabilityStatus.IMMEDIATE
    pricing_strategy: str = "reputation_premium"
    receipt_chain_length: int = 0
    success_rate: float = 0.0
    on_time_rate: Optional[float] = None
    strategy_consistency: float = 1.0
    endpoint: Optional[str] = None  # HTTP endpoint for handshake
    description: Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: Optional[str] = None  # ed25519 sig of content

    @property
    def is_newcomer(self) -> bool:
        """<20 receipts = newcomer, gets protected discovery slots."""
        return self.receipt_chain_length < 20

    @property
    def trust_tier(self) -> str:
        """Graduated trust per design doc onboarding sequence."""
        if self.receipt_chain_length < 10:
            return "newcomer"
        elif self.receipt_chain_length < 50:
            return "established"
        else:
            return "veteran"

    def content_hash(self) -> str:
        """Hash for signing (excludes signature field)."""
        d = asdict(self)
        d.pop("signature", None)
        d.pop("last_heartbeat", None)
        canonical = json.dumps(d, sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["availability"] = d["availability"]  # already string via StrEnum
        d["is_newcomer"] = self.is_newcomer
        d["trust_tier"] = self.trust_tier
        return d


@dataclass
class SearchQuery:
    """What a buyer searches for."""
    task_type: str
    verification_tier: int = 0
    pricing_strategy: Optional[str] = None
    min_success_rate: float = 0.0
    min_receipts: int = 0
    max_results: int = 10
    include_newcomers: bool = True  # if False, skip newcomer slots


@dataclass
class SearchResult:
    """A ranked match."""
    agent_id: str
    capability_ad: CapabilityAd
    relevance_score: float
    is_newcomer_slot: bool  # True if this result uses a protected slot


class Registry:
    """
    In-memory capability registry with newcomer protection.
    
    Discovery ranking algorithm:
    1. Filter by task_type match + verification_tier + availability
    2. Score: weighted sum of success_rate, strategy_consistency, on_time_rate, chain_length
    3. Reserve 30% of results for newcomers (Law 6)
    4. Sort by score within each slot type
    
    In production, this would be backed by a database + HTTP API.
    For the MVP, it's a pure Python in-memory registry.
    """

    NEWCOMER_RESERVE_FRACTION = 0.30  # 30% per Law 6
    HEARTBEAT_TIMEOUT_SECONDS = 3600  # 1 hour before marked stale

    def __init__(self):
        self.ads: dict[str, CapabilityAd] = {}  # agent_id -> ad
        self._platform_fee_rate = 0.05  # Start at 5%, cap at 15% (Law 7)

    def register(self, ad: CapabilityAd) -> bool:
        """Register or update a capability advertisement."""
        ad.last_heartbeat = datetime.now(timezone.utc).isoformat()
        self.ads[ad.agent_id] = ad
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """Update last-seen timestamp."""
        if agent_id not in self.ads:
            return False
        self.ads[agent_id].last_heartbeat = datetime.now(timezone.utc).isoformat()
        return True

    def deregister(self, agent_id: str) -> bool:
        """Remove from registry."""
        if agent_id in self.ads:
            del self.ads[agent_id]
            return True
        return False

    def _is_stale(self, ad: CapabilityAd) -> bool:
        """Check if agent hasn't heartbeated recently."""
        last = datetime.fromisoformat(ad.last_heartbeat)
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() > self.HEARTBEAT_TIMEOUT_SECONDS

    def _matches(self, ad: CapabilityAd, query: SearchQuery) -> bool:
        """Does this ad match the search query?"""
        if query.task_type not in ad.capabilities:
            return False
        if ad.verification_tier < query.verification_tier:
            return False
        if ad.availability in (AvailabilityStatus.BUSY, AvailabilityStatus.OFFLINE):
            return False
        if ad.success_rate < query.min_success_rate:
            return False
        if ad.receipt_chain_length < query.min_receipts:
            return False
        if query.pricing_strategy and ad.pricing_strategy != query.pricing_strategy:
            return False
        if self._is_stale(ad):
            return False
        return True

    def _score(self, ad: CapabilityAd) -> float:
        """
        Reputation-weighted score for ranking.
        
        Weights derived from simulation findings:
        - success_rate (40%): most direct quality signal
        - strategy_consistency (25%): Law 1 — switching destroys trust
        - on_time_rate (20%): reliability as Tier 0 reputation
        - chain_length_normalized (15%): experience, but diminishing returns
        """
        chain_norm = min(ad.receipt_chain_length / 100.0, 1.0)  # caps at 100
        on_time = ad.on_time_rate if ad.on_time_rate is not None else 0.5

        score = (
            0.40 * ad.success_rate +
            0.25 * ad.strategy_consistency +
            0.20 * on_time +
            0.15 * chain_norm
        )
        return round(score, 4)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        Search with newcomer-protected discovery slots.
        
        Algorithm:
        1. Find all matching ads
        2. Split into newcomers and veterans
        3. Reserve 30% of max_results for newcomers
        4. Fill remaining with veterans (by score)
        5. If not enough newcomers, backfill with veterans
        """
        # Find matches
        newcomer_matches = []
        veteran_matches = []

        for ad in self.ads.values():
            if not self._matches(ad, query):
                continue
            scored = SearchResult(
                agent_id=ad.agent_id,
                capability_ad=ad,
                relevance_score=self._score(ad),
                is_newcomer_slot=ad.is_newcomer,
            )
            if ad.is_newcomer:
                newcomer_matches.append(scored)
            else:
                veteran_matches.append(scored)

        # Sort each group by score (descending)
        newcomer_matches.sort(key=lambda r: r.relevance_score, reverse=True)
        veteran_matches.sort(key=lambda r: r.relevance_score, reverse=True)

        # Allocate slots
        total = query.max_results
        newcomer_slots = int(total * self.NEWCOMER_RESERVE_FRACTION) if query.include_newcomers else 0

        # Fill newcomer slots
        newcomer_results = newcomer_matches[:newcomer_slots]

        # Fill remaining with veterans
        veteran_slots = total - len(newcomer_results)
        veteran_results = veteran_matches[:veteran_slots]

        # Backfill: if not enough newcomers, give slots to veterans
        remaining = total - len(newcomer_results) - len(veteran_results)
        if remaining > 0:
            extra_veterans = veteran_matches[veteran_slots:veteran_slots + remaining]
            veteran_results.extend(extra_veterans)

        # Interleave: newcomers spread across results (not all at bottom)
        results = []
        v_idx, n_idx = 0, 0
        for i in range(total):
            if n_idx < len(newcomer_results) and (i % 3 == 0 or v_idx >= len(veteran_results)):
                results.append(newcomer_results[n_idx])
                n_idx += 1
            elif v_idx < len(veteran_results):
                results.append(veteran_results[v_idx])
                v_idx += 1

        return results

    def stats(self) -> dict:
        """Registry statistics."""
        total = len(self.ads)
        newcomers = sum(1 for a in self.ads.values() if a.is_newcomer)
        available = sum(1 for a in self.ads.values()
                       if a.availability == AvailabilityStatus.IMMEDIATE
                       and not self._is_stale(a))
        capabilities = set()
        for a in self.ads.values():
            capabilities.update(a.capabilities)

        return {
            "total_agents": total,
            "newcomers": newcomers,
            "veterans": total - newcomers,
            "available_now": available,
            "unique_capabilities": len(capabilities),
            "capabilities": sorted(capabilities),
            "newcomer_reserve": f"{self.NEWCOMER_RESERVE_FRACTION:.0%}",
            "platform_fee": f"{self._platform_fee_rate:.0%}",
        }

    def to_json(self) -> str:
        """Export registry as JSON (for static file serving in Era 1)."""
        return json.dumps({
            "registry_version": "0.1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.stats(),
            "agents": [ad.to_dict() for ad in self.ads.values()],
        }, indent=2)


# --- HTTP Registry Server (minimal, for Era 1) ---

def create_http_app(registry: Registry):
    """
    Create a minimal HTTP API for the registry.
    Uses Python's built-in http.server — no dependencies.
    
    Endpoints:
      GET  /registry          — full registry JSON
      GET  /registry/stats    — registry statistics
      POST /registry/register — register/update a capability ad
      POST /registry/search   — search for agents
      POST /registry/heartbeat — update last-seen
      DELETE /registry/<agent_id> — deregister
    
    In production, use FastAPI or similar. This is the MVP.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse

    class RegistryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/registry":
                self._json_response(200, json.loads(registry.to_json()))
            elif parsed.path == "/registry/stats":
                self._json_response(200, registry.stats())
            else:
                self._json_response(404, {"error": "not found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            body = self._read_body()

            if parsed.path == "/registry/register":
                try:
                    data = json.loads(body)
                    data["availability"] = AvailabilityStatus(data.get("availability", "immediate"))
                    ad = CapabilityAd(**data)
                    registry.register(ad)
                    self._json_response(200, {"ok": True, "agent_id": ad.agent_id})
                except Exception as e:
                    self._json_response(400, {"error": str(e)})

            elif parsed.path == "/registry/search":
                try:
                    data = json.loads(body)
                    query = SearchQuery(**data)
                    results = registry.search(query)
                    self._json_response(200, {
                        "results": [
                            {
                                "agent_id": r.agent_id,
                                "relevance_score": r.relevance_score,
                                "is_newcomer_slot": r.is_newcomer_slot,
                                "trust_tier": r.capability_ad.trust_tier,
                                "capabilities": r.capability_ad.capabilities,
                                "success_rate": r.capability_ad.success_rate,
                                "pricing_strategy": r.capability_ad.pricing_strategy,
                                "receipt_chain_length": r.capability_ad.receipt_chain_length,
                                "endpoint": r.capability_ad.endpoint,
                            }
                            for r in results
                        ],
                        "total": len(results),
                    })
                except Exception as e:
                    self._json_response(400, {"error": str(e)})

            elif parsed.path == "/registry/heartbeat":
                try:
                    data = json.loads(body)
                    ok = registry.heartbeat(data["agent_id"])
                    self._json_response(200, {"ok": ok})
                except Exception as e:
                    self._json_response(400, {"error": str(e)})
            else:
                self._json_response(404, {"error": "not found"})

        def do_DELETE(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.startswith("/registry/"):
                agent_id = parsed.path.split("/registry/", 1)[1]
                ok = registry.deregister(agent_id)
                self._json_response(200, {"ok": ok})
            else:
                self._json_response(404, {"error": "not found"})

        def _read_body(self) -> str:
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length).decode()

        def _json_response(self, status: int, data: dict):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, fmt, *args):
            pass  # suppress request logging

    return RegistryHandler


# --- Demo / Test ---

if __name__ == "__main__":
    import random

    registry = Registry()

    # Register a mix of newcomers and veterans
    task_types = ["code_review", "translation_en_zh", "data_transform",
                  "test_generation", "format_conversion", "api_integration"]

    agents = []
    for i in range(20):
        is_newcomer = i >= 14  # 6 newcomers, 14 veterans
        chain_len = random.randint(0, 15) if is_newcomer else random.randint(20, 150)
        success = random.uniform(0.6, 1.0) if chain_len > 0 else 0.0
        caps = random.sample(task_types, k=random.randint(1, 3))
        strategy = random.choice(["reputation_premium", "reputation_premium",
                                  "quality_premium", "market_rate"])

        ad = CapabilityAd(
            agent_id=f"ed25519:agent_{i:03d}",
            capabilities=caps,
            verification_tier=0,
            pricing_strategy=strategy,
            receipt_chain_length=chain_len,
            success_rate=round(success, 2),
            on_time_rate=round(random.uniform(0.7, 1.0), 2) if chain_len > 5 else None,
            strategy_consistency=round(random.uniform(0.7, 1.0), 2),
            endpoint=f"http://agent-{i:03d}.local:8080/handshake",
            description=f"Agent {i} specializing in {', '.join(caps)}",
        )
        registry.register(ad)
        agents.append(ad)

    # Show stats
    stats = registry.stats()
    print("=== Registry Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Search for code review agents
    print("\n=== Search: code_review ===")
    query = SearchQuery(task_type="code_review", max_results=5)
    results = registry.search(query)
    for i, r in enumerate(results):
        slot = "🆕" if r.is_newcomer_slot else "  "
        print(f"  {i+1}. {slot} {r.agent_id} | score={r.relevance_score:.3f} "
              f"| tier={r.capability_ad.trust_tier} | chain={r.capability_ad.receipt_chain_length} "
              f"| success={r.capability_ad.success_rate:.0%}")

    # Search with minimum requirements
    print("\n=== Search: translation, min 90% success, no newcomers ===")
    query2 = SearchQuery(
        task_type="translation_en_zh",
        min_success_rate=0.9,
        include_newcomers=False,
        max_results=3,
    )
    results2 = registry.search(query2)
    for i, r in enumerate(results2):
        print(f"  {i+1}. {r.agent_id} | score={r.relevance_score:.3f} "
              f"| success={r.capability_ad.success_rate:.0%} "
              f"| strategy={r.capability_ad.pricing_strategy}")

    # Export as static JSON
    print(f"\n=== Registry JSON size: {len(registry.to_json())} bytes ===")

    # Verify newcomer protection
    print("\n=== Newcomer Protection Check ===")
    query3 = SearchQuery(task_type="code_review", max_results=10)
    results3 = registry.search(query3)
    newcomer_count = sum(1 for r in results3 if r.is_newcomer_slot)
    total = len(results3)
    print(f"  Results: {total}, Newcomer slots: {newcomer_count} "
          f"({newcomer_count/max(total,1):.0%} of total, target=30%)")

    print("\n✓ Discovery MVP complete")
