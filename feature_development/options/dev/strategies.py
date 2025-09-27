from abc import ABC, abstractmethod
from optimize_leg_step_5 import optimize_leg

def optimize_strategy(df, strategy, top_n=5):
    candidates = strategy.generate_candidates(df)
    scored = [strategy.score_strategy(c) for c in candidates]
    scored.sort(key=lambda x: x["net_score"], reverse=True)
    return scored[:top_n]

class OptionStrategy(ABC):
    @abstractmethod
    def generate_candidates(self, df):
        pass

    @abstractmethod
    def score_strategy(self, legs):
        pass

class IronCondorStrategy(OptionStrategy):
    def generate_candidates(self, df):
        # Get candidate legs
        put_shorts = optimize_leg(df, option_type='P', position='Sell', top_n=3)
        put_longs  = optimize_leg(df, option_type='P', position='Buy', top_n=3)
        call_shorts = optimize_leg(df, option_type='C', position='Sell', top_n=3)
        call_longs  = optimize_leg(df, option_type='C', position='Buy', top_n=3)

        print("Put Shorts:", [p["strike"] for p in put_shorts])
        print("Put Longs:", [p["strike"] for p in put_longs])
        print("Call Shorts:", [c["strike"] for c in call_shorts])
        print("Call Longs:", [c["strike"] for c in call_longs])

        candidates = []
        for sp in put_shorts:
            # Only keep long puts lower than short put
            valid_longs_puts = [lp for lp in put_longs if lp["strike"] < sp["strike"]]
            for lp in valid_longs_puts:
                for sc in call_shorts:
                    # Only keep long calls higher than short call
                    valid_longs_calls = [lc for lc in call_longs if lc["strike"] > sc["strike"]]
                    for lc in valid_longs_calls:
                        candidates.append({
                            "short_put": sp,
                            "long_put": lp,
                            "short_call": sc,
                            "long_call": lc
                        })
        return candidates


    def score_strategy(self, legs):
        # Aggregate leg scores
        net_score = (
            legs["short_put"]["score"]
            + legs["long_put"]["score"]
            + legs["short_call"]["score"]
            + legs["long_call"]["score"]
        )
        return {
            "legs": legs,
            "net_score": net_score,
            "net_delta": (
                legs["short_put"]["delta"] + legs["long_put"]["delta"]
                + legs["short_call"]["delta"] + legs["long_call"]["delta"]
            ),
            "net_theta": (
                legs["short_put"]["theta"] + legs["long_put"]["theta"]
                + legs["short_call"]["theta"] + legs["long_call"]["theta"]
            ),
            "net_vega": (
                legs["short_put"]["vega"] + legs["long_put"]["vega"]
                + legs["short_call"]["vega"] + legs["long_call"]["vega"]
            )
        }
