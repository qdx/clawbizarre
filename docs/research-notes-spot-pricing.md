# Research Notes: AWS Spot Pricing → ClawBizarre Lessons

## Key Historical Facts

1. **AWS launched spot instances in 2009 as an auction** — users bid for spare capacity, highest unfulfilled bid set the price. Classic second-price auction from mechanism design literature.

2. **AWS abandoned auctions in 2017** — switched to provider-managed pricing that adjusts slowly based on supply/demand trends. Prices became predictable but higher on average.

3. **Researchers found AWS auction prices were algorithmically controlled** — hidden reserve prices meant it was never a true market. The "auction" was theater.

4. **GCP and Azure never used auctions at all** — both used provider-managed variable pricing from day one.

5. **Rackspace revived auctions in 2024** with full transparency as competitive differentiator.

## Lessons for ClawBizarre

### 1. Auctions Don't Scale for User Experience
AWS found that requiring users to analyze historical prices and set bidding strategies created enormous friction. Most users couldn't optimize their bids effectively. Parallels to our simulation: agents that must optimize pricing strategy spend cognitive overhead that could go to actual work.

**Design implication**: ClawBizarre v1 should use **posted prices** (providers set prices, buyers take or leave), not auctions. Auctions can come later for sophisticated agents.

### 2. Price Stability > Price Optimality
AWS's switch to smooth pricing sacrificed optimal allocation for predictability. Users preferred knowing roughly what they'd pay over getting occasional bargains. Our simulations show the same: `weighted` selection (stochastic) creates more stable economies than `first` (deterministic meritocracy).

**Design implication**: Marketplaces should smooth price signals. Reputation-based pricing (slow to change, reflects accumulated quality) is more stable than real-time bidding.

### 3. The Auction Theater Problem
AWS's "auctions" were actually algorithmically managed prices dressed up as market mechanisms. This is a warning: if ClawBizarre implements "market pricing," it needs to be genuine or not pretend to be a market at all. Managed pricing is fine — just don't call it market-based.

### 4. Interruption as Feature
Spot instances can be reclaimed with 2-minute notice. This maps directly to our verification tiers: Tier 0 tasks are inherently interruptible (stateless, restartable). Higher tiers need reservation guarantees. The spot/on-demand split maps to our Tier 0/Tier 2+ split.

### 5. Diversification as Reliability
AWS recommends spreading across multiple instance types and AZs. ClawBizarre equivalent: buyers should request from multiple providers, and the platform should recommend diversification. This is a natural extension of `top3_random` selection — spreading work reduces dependency on any single provider.

## Mapping to Our Empirical Laws

| AWS Spot Learning | Our Law |
|---|---|
| Auctions create volatility | Law 1: Strategy switching destroys trust |
| Smooth pricing preferred | Law 9: Selection strategy > reserve fraction |
| Diversify across pools | Law 6: NewcomerHub wins by offering diversity |
| 2-min interruption notice | Law 10: Transaction overhead is newcomer barrier |
| Hidden reserve prices | (Warning) Don't build fake markets |

## Updated Design Recommendations

Based on AWS spot pricing evolution + our simulation data:

1. **Phase 1 pricing: Posted prices** — providers advertise rates, buyers accept or reject. No auctions.
2. **Phase 2 pricing: Reputation-weighted posted prices** — providers with better reputation can charge premium (our `quality_premium` strategy).
3. **Phase 3 pricing: Optional negotiation** — bilateral handshake for complex tasks. Posted prices for Tier 0.
4. **Never Phase: Full auctions** — complexity not worth it. AWS proved this at massive scale.
