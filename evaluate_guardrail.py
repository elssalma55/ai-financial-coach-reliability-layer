"""Evaluate the financial guardrail using reproducible synthetic scenarios."""
import copy
import csv
from pathlib import Path
from financial_validator import validate_financial_feasibility

BASE_DATA = {
    "monthly_income": 3000,
    "manual_expenses": {"Housing": 1600, "Food": 400},
    "debts": [{"amount": 1000, "min_payment": 100}],
}
BASE_RESULTS = {
    "budget_analysis": {"total_expenses": 2000},
    "savings_strategy": {"recommendations": [{"amount": 400}]},
    "debt_reduction": {"total_debt": 1000},
}


def case(name, change, expected):
    data = copy.deepcopy(BASE_DATA)
    results = copy.deepcopy(BASE_RESULTS)
    change(data, results)
    return name, data, results, expected


CASES = [
    case("Valid baseline", lambda d, r: None, "PASS"),
    case("Valid low savings", lambda d, r: r["savings_strategy"]["recommendations"][0].update(amount=100), "PASS"),
    case("Valid at cash-flow limit", lambda d, r: r["savings_strategy"]["recommendations"][0].update(amount=900), "PASS"),
    case("No debt valid", lambda d, r: (d.update(debts=[]), r["debt_reduction"].update(total_debt=0)), "PASS"),
    case("Unrealistic savings", lambda d, r: r["savings_strategy"]["recommendations"][0].update(amount=1100), "FAIL"),
    case("Wrong expenses", lambda d, r: r["budget_analysis"].update(total_expenses=1900), "FAIL"),
    case("Wrong debt total", lambda d, r: r["debt_reduction"].update(total_debt=800), "FAIL"),
    case("Invalid savings amount", lambda d, r: r["savings_strategy"]["recommendations"][0].update(amount="bad"), "FAIL"),
    case("Negative savings", lambda d, r: r["savings_strategy"]["recommendations"][0].update(amount=-100), "FAIL"),
    case("Invalid income", lambda d, r: d.update(monthly_income="bad"), "FAIL"),
    case("Missing expense total", lambda d, r: r["budget_analysis"].pop("total_expenses"), "WARNING"),
    case("Negative cash flow alone", lambda d, r: (d.update(monthly_income=1800), r["savings_strategy"]["recommendations"][0].update(amount=0)), "WARNING"),
]


def run():
    rows = []
    for name, data, outputs, expected in CASES:
        actual = validate_financial_feasibility(data, outputs)["status"]
        rows.append({"test": name, "expected": expected, "actual": actual,
                     "passed": actual == expected})
    passed = sum(row["passed"] for row in rows)
    print("Financial Validation Layer — Reproducible Evaluation")
    for row in rows:
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['test']}: {row['actual']} (expected: {row['expected']})")
    print(f"\n{passed}/{len(rows)} cases passed ({passed / len(rows) * 100:.1f}% test success rate)")
    output = Path(__file__).with_name("evaluation_results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test", "expected", "actual", "passed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results exported to: {output.name}")
    return passed == len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
