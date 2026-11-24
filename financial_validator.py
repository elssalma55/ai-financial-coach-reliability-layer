"""Simple financial checks used after the AI agents finish."""
import math


def _number(value):
    """Return a finite number, or None when the input is not usable."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def validate_financial_feasibility(data, results):
    findings = []
    income = _number(data.get("monthly_income"))
    transactions = data.get("transactions") or []
    manual_expenses = data.get("manual_expenses") or {}
    debts = data.get("debts") or []

    # Recalculate the source totals rather than trusting model outputs
    amounts = ([item.get("Amount") for item in transactions] if transactions
               else list(manual_expenses.values()))
    checked_amounts = [_number(value) for value in amounts]
    if income is None or any(value is None for value in checked_amounts):
        return {"status": "FAIL", "available_cash_flow": 0.0,
                "recommended_savings": 0.0,
                "findings": [{"status": "FAIL", "message": "Income or expense amount is invalid."}]}
    expenses = sum(checked_amounts)

    payments = [_number(debt.get("min_payment") or 0) for debt in debts]
    if any(value is None or value < 0 for value in payments):
        findings.append({"status": "FAIL", "message": "A debt minimum payment is invalid."})
    minimum_payments = sum(value for value in payments if value is not None)
    available = income - expenses - minimum_payments

    # Recommendations are assumed to be additional monthly allocations
    recommendations = (results.get("savings_strategy") or {}).get("recommendations") or []
    savings_values = [_number(rec.get("amount")) for rec in recommendations]
    if any(value is None or value < 0 for value in savings_values):
        findings.append({"status": "FAIL", "message": "A savings recommendation has an invalid amount."})
    proposed_savings = sum(value for value in savings_values if value is not None)

    if available < -0.01:
        findings.append({"status": "WARNING", "message": "Monthly cash flow is negative."})
    if proposed_savings > max(available, 0) + 0.01:
        findings.append({"status": "FAIL", "message":
            f"Suggested savings (${proposed_savings:,.2f}) exceed available cash flow (${available:,.2f})."})

    reported = _number((results.get("budget_analysis") or {}).get("total_expenses"))
    if reported is None:
        findings.append({"status": "WARNING", "message": "The AI did not provide a valid expense total."})
    elif abs(reported - expenses) > 0.01:
        findings.append({"status": "FAIL", "message":
            f"AI expense total (${reported:,.2f}) differs from input total (${expenses:,.2f})."})

    # Check the debt balance reported by the AI against input data
    debt_balances = [_number(debt.get("amount")) for debt in debts]
    reported_debt = _number((results.get("debt_reduction") or {}).get("total_debt"))
    if any(value is None for value in debt_balances):
        findings.append({"status": "FAIL", "message": "A debt balance is invalid."})
    elif reported_debt is not None and abs(reported_debt - sum(debt_balances)) > 0.01:
        findings.append({"status": "FAIL", "message": "AI total debt differs from the input balances."})

    status = "FAIL" if any(f["status"] == "FAIL" for f in findings) else (
        "WARNING" if findings else "PASS")
    return {"status": status, "available_cash_flow": round(available, 2),
            "recommended_savings": round(proposed_savings, 2), "findings": findings}
