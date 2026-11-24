"""Basic checks for the new financial validation layer."""
import unittest
from financial_validator import validate_financial_feasibility


class FinancialValidatorTests(unittest.TestCase):
    def setUp(self):
        self.data = {"monthly_income": 3000, "manual_expenses": {"Housing": 1600, "Food": 400},
                     "debts": [{"amount": 1000, "min_payment": 100}]}
        self.results = {"budget_analysis": {"total_expenses": 2000},
                        "savings_strategy": {"recommendations": [{"amount": 400}]},
                        "debt_reduction": {"total_debt": 1000}}

    def test_valid_advice(self):
        self.assertEqual(validate_financial_feasibility(self.data, self.results)["status"], "PASS")

    def test_unaffordable_savings(self):
        self.results["savings_strategy"]["recommendations"][0]["amount"] = 1100
        self.assertEqual(validate_financial_feasibility(self.data, self.results)["status"], "FAIL")

    def test_expense_mismatch(self):
        self.results["budget_analysis"]["total_expenses"] = 1900
        self.assertEqual(validate_financial_feasibility(self.data, self.results)["status"], "FAIL")

    def test_negative_cash_flow(self):
        self.data["monthly_income"] = 1800
        self.assertEqual(validate_financial_feasibility(self.data, self.results)["status"], "FAIL")

    def test_debt_mismatch(self):
        self.results["debt_reduction"]["total_debt"] = 800
        self.assertEqual(validate_financial_feasibility(self.data, self.results)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
