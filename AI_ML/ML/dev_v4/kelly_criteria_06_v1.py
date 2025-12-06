# ==============================================================
#                 KELLY CRITERION WITH FOS + SAFETY
# ==============================================================

def kelly_with_fos(
    capital,
    leverage,
    win_prob,
    avg_win,
    avg_loss,
    fos=2,                  # Factor of Safety (2=Half Kelly, 4=Quarter Kelly)
    max_risk_frac=0.02      # Hard safety cap, e.g., 2%
):
    """
    capital       : total capital (float)
    leverage      : MIS leverage multiplier (1 to 5)
    win_prob      : probability of winning (0–1)
    avg_win       : average win (decimal)
    avg_loss      : average loss (decimal)
    fos           : factor of safety (bigger = safer)
    max_risk_frac : maximum fraction of capital you allow risking
    """

    p = win_prob
    q = 1 - p

    # Payoff ratio (how much you win for every 1 unit you lose)
    b = avg_win / avg_loss

    # --- Raw Kelly formula ---
    kelly_raw = p - (q / b)

    # If the system is unprofitable → no bet
    if kelly_raw <= 0:
        return {
            "kelly_raw": 0,
            "kelly_fos": 0,
            "kelly_after_leverage": 0,
            "final_risk_fraction": 0,
            "risk_amount": 0,
            "note": "❌ Negative expectancy — no trade recommended."
        }

    # --- Apply Factor of Safety ---
    kelly_fos = kelly_raw / fos

    # --- Reduce position due to leverage ---
    kelly_after_leverage = kelly_fos / leverage

    # --- Apply hard safety cap ---
    final_risk_fraction = min(kelly_after_leverage, max_risk_frac)

    # --- Convert to capital amount ---
    risk_amount = capital * final_risk_fraction

    return {
        "kelly_raw": kelly_raw,
        "kelly_fos": kelly_fos,
        "kelly_after_leverage": kelly_after_leverage,
        "final_risk_fraction": final_risk_fraction,
        "risk_amount": risk_amount,
        "note": "✔ Kelly reduced by FOS, leverage, and safety cap."
    }



# ==============================================================
#                 BEAUTIFUL PRINTING FUNCTION
# ==============================================================

def pretty_print(result):
    print("\n" + "="*55)
    print("                   📌 KELLY OUTPUT")
    print("="*55)

    print(f"🔸 Raw Kelly Fraction           : {result['kelly_raw']:.4f}")
    print(f"🔸 After FOS (Safety Applied)   : {result['kelly_fos']:.4f}")
    print(f"🔸 After Leverage Adjustment    : {result['kelly_after_leverage']:.4f}")

    print("-"*55)
    print(f"🔸 Final Risk Fraction (capped) : {result['final_risk_fraction']:.4f}")
    print(f"💰 Risk Amount per Trade        : {result['risk_amount']:.2f}")
    print("-"*55)

    print(f"📘 Note: {result['note']}")
    print("="*55 + "\n")



# ==============================================================
#                 EXAMPLE USAGE
# ==============================================================

capital   = 100000
leverage  = 5
win_prob  = 0.7534
avg_win   = 0.016224
avg_loss  = 0.008993
fos       = 1             # quarter Kelly
max_risk_frac = 0.5      # 2% maximum risk

result = kelly_with_fos(capital, leverage, win_prob, avg_win, avg_loss, fos, max_risk_frac)
pretty_print(result)
