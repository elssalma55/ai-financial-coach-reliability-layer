"""Deterministic sample outputs for UI and validator integration testing.

This module does NOT execute any LLM agents. It provides complete mock outputs
matching the original agents' response shapes so all dashboard charts can be tested.
"""
from collections import defaultdict


def create_demo_results(financial_data, case="Unrealistic savings recommendation"):
    transactions = financial_data.get("transactions") or []
    manual = financial_data.get("manual_expenses") or {}
    categories = defaultdict(float)
    if transactions:
        for transaction in transactions:
            categories[str(transaction["Category"])] += float(transaction["Amount"])
    else:
        for category, amount in manual.items():
            if float(amount):
                categories[str(category)] += float(amount)

    expenses = sum(categories.values())
    income = float(financial_data.get("monthly_income") or 0)
    debts = financial_data.get("debts") or []
    debt_total = sum(float(d["amount"]) for d in debts)
    payments = sum(float(d.get("min_payment") or 0) for d in debts)
    cash_flow = income - expenses - payments

    savings = max(0.0, cash_flow * 0.25)
    reported_expenses = expenses
    if case == "Unrealistic savings recommendation":
        savings = max(100.0, cash_flow + 200.0)
    elif case == "Incorrect AI expense total":
        reported_expenses += 100.0

    if case == "Incorrect AI expense total":
        finding = ("FAIL", "The expense total is inconsistent with the uploaded CSV.",
                   "Check the underlying category totals.")
    elif savings > max(0.0, cash_flow):
        finding = ("FAIL", "The savings target exceeds available monthly cash flow.",
                   "Lower the savings target and prioritize essential spending.")
    elif cash_flow < 0:
        finding = ("WARNING", "Monthly cash flow is negative.",
                   "Review essential expenses and debt obligations.")
    else:
        finding = ("PASS", "No inconsistency in this synthetic scenario.",
                   "Continue monitoring the budget.")

    def payoff(method):
        # The sample is illustrative, not a simulated amortization schedule.
        return {"total_interest": round(debt_total * (0.2 if method == "avalanche" else 0.25), 2),
                "months_to_payoff": 24 if debt_total else 0,
                "monthly_payment": round(debt_total / 24, 2)}

    return {
        "budget_analysis": {
            "total_expenses": round(reported_expenses, 2),
            "monthly_income": income,
            "spending_categories": [
                {"category": c, "amount": round(amount, 2),
                 "percentage": round(100 * amount / expenses, 2) if expenses else 0}
                for c, amount in categories.items()
            ],
            "recommendations": [{
                "category": "Budget review", "recommendation": "Review larger spending categories first.",
                "potential_savings": round(expenses * 0.05, 2)
            }],
        },
        "savings_strategy": {
            "emergency_fund": {
                "recommended_amount": round(expenses * 3, 2), "current_amount": 0.0,
                "current_status": "Example scenario — not assessed by an LLM"
            },
            "recommendations": [{
                "category": "Example monthly allocation", "amount": round(savings, 2),
                "rationale": "Synthetic recommendation used to test the audit layer."
            }],
            "automation_techniques": [{
                "name": "Recurring budget review",
                "description": "Review expenses on a regular schedule."
            }],
        },
        "debt_reduction": {
            "total_debt": round(debt_total, 2),
            "debts": debts,
            "payoff_plans": {
                "avalanche": payoff("avalanche"),
                "snowball": payoff("snowball"),
            },
            "recommendations": [{
                "title": "Check repayment affordability",
                "description": "Review minimum payments against monthly cash flow.",
                "impact": "Test scenario only; payoff figures are illustrative."
            }],
        },
        "financial_audit": {
            "overall_status": finding[0],
            "findings": [{"severity": finding[0], "issue": finding[1],
                          "recommendation": finding[2]}],
            "summary": "Synthetic audit example for integration testing, not an AI-generated assessment."
        }
    }
