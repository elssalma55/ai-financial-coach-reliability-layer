import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO
from uuid import uuid4
from financial_validator import validate_financial_feasibility
from full_demo_results import create_demo_results

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from ollama_structured_llm import OllamaStructuredLlm
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

# Pydantic models for output schemas
class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

# Output format for the new audit agent
class AuditFinding(BaseModel):
    severity: str = Field(..., description="PASS, WARNING or FAIL")
    issue: str = Field(..., description="Financial issue found")
    recommendation: str = Field(..., description="Action to consider")

class FinancialAuditReport(BaseModel):
    overall_status: str = Field(..., description="PASS, WARNING or FAIL")
    findings: List[AuditFinding] = Field(default_factory=list)
    summary: str = Field(..., description="Short review of the recommendations")

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def make_llm(schema_type):
    # Enforce each agent's existing Pydantic schema in Ollama
    return OllamaStructuredLlm(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        schema_type=schema_type,
    )

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """Safely parse JSON data with error handling"""
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

class FinanceAdvisorSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()
        
        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model=make_llm(BudgetAnalysis),
            description="Analyzes financial data to categorize spending patterns and recommend budget improvements",
            instruction="""You are a Budget Analysis Agent specialized in reviewing financial transactions and expenses.
You are the first agent in a sequence of financial advisor agents.

Your tasks:
1. Analyze income, transactions, and expenses in detail
2. Categorize spending into logical groups with clear breakdown
3. Identify spending patterns and trends across categories
4. Suggest specific areas where spending could be reduced with concrete suggestions
5. Provide actionable recommendations with specific, quantified potential savings amounts

Consider:
- Number of dependants when evaluating household expenses
- Typical spending ratios for the income level (housing 30%, food 15%, etc.)
- Essential vs discretionary spending with clear separation
- Seasonal spending patterns if data spans multiple months

For spending categories, include ALL expenses from the user's data, ensure percentages add up to 100%,
and make sure every expense is categorized.

For recommendations:
- Provide at least 3-5 specific, actionable recommendations with estimated savings
- Explain the reasoning behind each recommendation
- Consider the impact on quality of life and long-term financial health
- Suggest specific implementation steps for each recommendation

IMPORTANT: Store your analysis in state['budget_analysis'] for use by subsequent agents.""",
            output_schema=BudgetAnalysis,
            output_key="budget_analysis"
        )
        
        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model=make_llm(SavingsStrategy),
            description="Recommends optimal savings strategies based on income, expenses, and financial goals",
            instruction="""You are a Savings Strategy Agent specialized in creating personalized savings plans.
You are the second agent in the sequence. READ the budget analysis from state['budget_analysis'] first.

Your tasks:
1. Review the budget analysis results from state['budget_analysis']
2. Recommend comprehensive savings strategies based on the analysis
3. Calculate optimal emergency fund size based on expenses and dependants
4. Suggest appropriate savings allocation across different purposes
5. Recommend practical automation techniques for saving consistently

Consider:
- Risk factors based on job stability and dependants
- Balancing immediate needs with long-term financial health
- Progressive savings rates as discretionary income increases
- Multiple savings goals (emergency, retirement, specific purchases)
- Areas of potential savings identified in the budget analysis

IMPORTANT: Store your strategy in state['savings_strategy'] for use by the Debt Reduction Agent.""",
            output_schema=SavingsStrategy,
            output_key="savings_strategy"
        )
        
        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model=make_llm(DebtReduction),
            description="Creates optimized debt payoff plans to minimize interest paid and time to debt freedom",
            instruction="""You are a Debt Reduction Agent specialized in creating debt payoff strategies.
You are the third agent in the sequence. READ both state['budget_analysis'] and state['savings_strategy'] first.

Your tasks:
1. Review both budget analysis and savings strategy from the state
2. Analyze debts by interest rate, balance, and minimum payments
3. Create prioritized debt payoff plans (avalanche and snowball methods)
4. Calculate total interest paid and time to debt freedom
5. Suggest debt consolidation or refinancing opportunities
6. Provide specific recommendations to accelerate debt payoff

Consider:
- Cash flow constraints from the budget analysis
- Emergency fund and savings goals from the savings strategy
- Psychological factors (quick wins vs mathematical optimization)
- Credit score impact and improvement opportunities

IMPORTANT: Store your final plan in state['debt_reduction'] and ensure it aligns with the previous analyses.""",
            output_schema=DebtReduction,
            output_key="debt_reduction"
        )
        
        # Fourth agent: review the other agents before showing the advice
        self.financial_audit_agent = LlmAgent(
            name="FinancialAuditAgent",
            model=make_llm(FinancialAuditReport),
            description="Reviews the financial advice for inconsistencies and risks",
            instruction="""You are a financial quality review agent.
You run after the budget, savings, and debt agents.

Review the original user data and these session outputs:
- state['budget_analysis']
- state['savings_strategy']
- state['debt_reduction']

Look for conflicting advice, unrealistic savings targets, and debt recommendations
that do not consider the user's available cash flow.
Do not invent financial inputs or claim your audit guarantees correctness.
The Python validation layer independently checks the numerical constraints.
Return a short report with PASS, WARNING or FAIL and clear findings.
""",
            output_schema=FinancialAuditReport,
            output_key="financial_audit"
        )

        self.coordinator_agent = SequentialAgent(
            name="FinanceCoordinatorAgent",
            description="Coordinates specialized finance agents to provide comprehensive financial advice",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent,
                self.financial_audit_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"finance_session_{uuid4().hex}"
        
        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions", []),
                "manual_expenses": financial_data.get("manual_expenses", {}),
                "debts": financial_data.get("debts", [])
            }
            
            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )
            
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)
            
            default_results = self._create_default_results(financial_data)
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data))]
            )
            
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
            
            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]

            # Keep the audit separate so we never mistake a fallback for an AI review
            audit_value = updated_session.state.get("financial_audit")
            results["financial_audit"] = parse_json_safely(audit_value) if audit_value else None
            return results
            
        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            raise
        finally:
            self.session_service.delete_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )
    
    def _preprocess_transactions(self, session):
        transactions = session.state.get("transactions", [])
        if not transactions:
            return
        
        df = pd.DataFrame(transactions)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        if 'Category' in df.columns and 'Amount' in df.columns:
            category_spending = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["category_spending"] = category_spending
            session.state["total_spending"] = df['Amount'].sum()
    
    def _preprocess_manual_expenses(self, session):
        manual_expenses = session.state.get("manual_expenses", {})
        if not manual_expenses or manual_expenses is None:
            return
        
        session.state.update({
            "total_manual_spending": sum(manual_expenses.values()),
            "manual_category_spending": manual_expenses
        })

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        monthly_income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses", {})
        
        # Ensure expenses is not None
        if expenses is None:
            expenses = {}
        
        if not expenses and financial_data.get("transactions"):
            expenses = {}
            for transaction in financial_data["transactions"]:
                category = transaction.get("Category", "Uncategorized")
                amount = transaction.get("Amount", 0)
                expenses[category] = expenses.get(category, 0) + amount
        
        total_expenses = sum(expenses.values())
        
        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": monthly_income,
                "spending_categories": [
                    {"category": cat, "amount": amt, "percentage": (amt / total_expenses * 100) if total_expenses > 0 else 0}
                    for cat, amt in expenses.items()
                ],
                "recommendations": [
                    {"category": "General", "recommendation": "Consider reviewing your expenses carefully", "potential_savings": total_expenses * 0.1}
                ]
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Not started"
                },
                "recommendations": [
                    {"category": "Emergency Fund", "amount": total_expenses * 0.1, "rationale": "Build emergency fund first"},
                    {"category": "Retirement", "amount": monthly_income * 0.15, "rationale": "Long-term savings"}
                ],
                "automation_techniques": [
                    {"name": "Automatic Transfer", "description": "Set up automatic transfers on payday"}
                ]
            },
            "debt_reduction": {
                "total_debt": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])),
                "debts": financial_data.get("debts", []),
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    },
                    "snowball": {
                        "total_interest": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": sum(debt.get("amount", 0) for debt in financial_data.get("debts", [])) / 24
                    }
                },
                "recommendations": [
                    {"title": "Increase Payments", "description": "Increase your monthly payments", "impact": "Reduces total interest paid"}
                ]
            }
        }

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis results")
            return
    
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format")
        return
    
    if "spending_categories" in analysis:
        st.subheader("Spending by Category")
        fig = px.pie(
            values=[cat["amount"] for cat in analysis["spending_categories"]],
            names=[cat["category"] for cat in analysis["spending_categories"]],
            title="Your Spending Breakdown"
        )
        st.plotly_chart(fig)
    
    if "total_expenses" in analysis:
        st.subheader("Income vs. Expenses")
        income = analysis.get("monthly_income", 0)
        expenses = analysis["total_expenses"]
        surplus_deficit = income - expenses
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Income", "Expenses"], 
                            y=[income, expenses],
                            marker_color=["green", "red"]))
        fig.update_layout(title="Monthly Income vs. Expenses")
        st.plotly_chart(fig)
        
        st.metric("Monthly Surplus/Deficit", 
                  f"${surplus_deficit:.2f}", 
                  delta=f"{surplus_deficit:.2f}")
    
    if "recommendations" in analysis:
        st.subheader("Spending Reduction Recommendations")
        for rec in analysis["recommendations"]:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            if "potential_savings" in rec:
                st.metric(f"Potential Monthly Savings", f"${rec['potential_savings']:.2f}")

def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy results")
            return
    
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format")
        return
    
    st.subheader("Savings Recommendations")
    
    if "emergency_fund" in strategy:
        ef = strategy["emergency_fund"]
        st.markdown(f"### Emergency Fund")
        st.markdown(f"**Recommended Size**: ${ef['recommended_amount']:.2f}")
        st.markdown(f"**Current Status**: {ef['current_status']}")
        
        if ef.get("current_amount") is not None and (ef.get("recommended_amount") or 0) > 0:
            progress = ef["current_amount"] / ef["recommended_amount"]
            st.progress(min(progress, 1.0))
            st.markdown(f"${ef['current_amount']:.2f} of ${ef['recommended_amount']:.2f}")
    
    if "recommendations" in strategy:
        st.markdown("### Recommended Savings Allocations")
        for rec in strategy["recommendations"]:
            st.markdown(f"**{rec['category']}**: ${rec['amount']:.2f}/month")
            st.markdown(f"_{rec['rationale']}_")
    
    if "automation_techniques" in strategy:
        st.markdown("### Automation Techniques")
        for technique in strategy["automation_techniques"]:
            st.markdown(f"**{technique['name']}**: {technique['description']}")

def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction results")
            return
    
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format")
        return
    
    if "total_debt" in plan:
        st.metric("Total Debt", f"${plan['total_debt']:.2f}")
    
    if "debts" in plan:
        st.subheader("Your Debts")
        debt_df = pd.DataFrame(plan["debts"])
        st.dataframe(debt_df)
        
        if not debt_df.empty:
            fig = px.bar(debt_df, x="name", y="amount", color="interest_rate",
                        labels={"name": "Debt", "amount": "Amount ($)", "interest_rate": "Interest Rate (%)"},
                        title="Debt Breakdown")
            st.plotly_chart(fig)
    
    if "payoff_plans" in plan:
        st.subheader("Debt Payoff Plans")
        tabs = st.tabs(["Avalanche Method", "Snowball Method", "Comparison"])
        
        with tabs[0]:
            st.markdown("### Avalanche Method (Highest Interest First)")
            if "avalanche" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                st.markdown(f"**Total Interest Paid**: ${avalanche['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {avalanche['months_to_payoff']} months")
                
                if "monthly_payment" in avalanche:
                    st.markdown(f"**Recommended Monthly Payment**: ${avalanche['monthly_payment']:.2f}")
        
        with tabs[1]:
            st.markdown("### Snowball Method (Smallest Balance First)")
            if "snowball" in plan["payoff_plans"]:
                snowball = plan["payoff_plans"]["snowball"]
                st.markdown(f"**Total Interest Paid**: ${snowball['total_interest']:.2f}")
                st.markdown(f"**Time to Debt Freedom**: {snowball['months_to_payoff']} months")
                
                if "monthly_payment" in snowball:
                    st.markdown(f"**Recommended Monthly Payment**: ${snowball['monthly_payment']:.2f}")
        
        with tabs[2]:
            st.markdown("### Method Comparison")
            if "avalanche" in plan["payoff_plans"] and "snowball" in plan["payoff_plans"]:
                avalanche = plan["payoff_plans"]["avalanche"]
                snowball = plan["payoff_plans"]["snowball"]
                
                comparison_data = {
                    "Method": ["Avalanche", "Snowball"],
                    "Total Interest": [avalanche["total_interest"], snowball["total_interest"]],
                    "Months to Payoff": [avalanche["months_to_payoff"], snowball["months_to_payoff"]]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                st.dataframe(comparison_df)
                
                fig = go.Figure(data=[
                    go.Bar(name="Total Interest", x=comparison_df["Method"], y=comparison_df["Total Interest"]),
                    go.Bar(name="Months to Payoff", x=comparison_df["Method"], y=comparison_df["Months to Payoff"])
                ])
                fig.update_layout(barmode='group', title="Debt Payoff Method Comparison")
                st.plotly_chart(fig)
    
    if "recommendations" in plan:
        st.subheader("Debt Reduction Recommendations")
        for rec in plan["recommendations"]:
            st.markdown(f"**{rec['title']}**: {rec['description']}")
            if "impact" in rec:
                st.markdown(f"_Impact: {rec['impact']}_")

def display_financial_audit(validation: Dict[str, Any], agent_audit: Optional[Dict[str, Any]]):
    """Show deterministic checks and the AI review as separate results."""
    st.subheader("Financial Quality Audit")
    st.metric("Available cash flow", f"${validation['available_cash_flow']:,.2f}")
    st.metric("Suggested monthly savings (if available)", f"${validation['recommended_savings']:,.2f}")

    # Check how much the user can currently afford to save
    available = validation["available_cash_flow"]
    suggested = validation["recommended_savings"]
    safe_limit = max(0.0, available)

    st.subheader("Safe Financial Recommendation")
    st.metric("Affordable Savings Limit", f"${safe_limit:,.2f}")

    if available < 0:
        st.warning(
            f"Monthly cash flow is negative by ${abs(available):,.2f}. "
            "Reduce the deficit before allocating money to savings."
        )
    elif suggested > safe_limit:
        st.warning(
            f"AI suggested ${suggested:,.2f}, but only "
            f"${safe_limit:,.2f} is currently available."
        )
    else:
        st.success("Suggested savings are within the available cash flow.")

    st.write(f"**Rule-based validation: {validation['status']}**")
    for finding in validation["findings"]:
        if finding["status"] == "FAIL":
            st.error(finding["message"].replace("$", r"\$"))
        else:
            st.warning(finding["message"].replace("$", r"\$"))
    if not validation["findings"]:
        st.success("No issues detected by the current financial rules.")

    st.markdown("#### AI Audit Agent")
    if isinstance(agent_audit, dict):
        st.write(f"**Agent assessment: {agent_audit.get('overall_status', 'UNKNOWN')}**")
        st.write(agent_audit.get("summary", "No summary available."))
        for item in agent_audit.get("findings") or []:
            st.write(f"**{item.get('severity', 'INFO')}** — {item.get('issue', '')}")
            st.caption(item.get("recommendation", ""))
    else:
        st.info("The AI audit result is not available. Rule-based checks are shown above.")


def parse_csv_transactions(file_content) -> List[Dict[str, Any]]:
    """Parse CSV file content into a list of transactions"""
    try:
        # Read CSV content
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date strings to datetime and then to string format YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Convert amount strings to float, handling currency symbols and commas
        df['Amount'] = df['Amount'].replace(r'[$,]', '', regex=True).astype(float)
        
        # Group by category and calculate totals
        category_totals = df.groupby('Category')['Amount'].sum().reset_index()
        
        # Convert to list of dictionaries
        transactions = df.to_dict('records')
        
        return {
            'transactions': transactions,
            'category_totals': category_totals.to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def validate_csv_format(file) -> bool:
    """Validate CSV file format and content"""
    try:
        content = file.read().decode('utf-8')
        dialect = csv.Sniffer().sniff(content)
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)  # Reset file pointer
        
        if not has_header:
            return False, "CSV file must have headers"
            
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
            
        # Validate date format
        try:
            pd.to_datetime(df['Date'])
        except Exception:
            return False, "Invalid date format in Date column"
            
        # Validate amount format (should be numeric after removing currency symbols)
        try:
            df['Amount'].replace(r'[$,]', '', regex=True).astype(float)
        except Exception:
            return False, "Invalid amount format in Amount column"
            
        return True, "CSV format is valid"
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"

def display_csv_preview(df: pd.DataFrame):
    """Display a preview of the CSV data with basic statistics"""
    st.subheader("CSV Data Preview")
    
    # Show basic statistics
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    
    # Convert dates for display
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%Y-%m-%d')} to {df_dates.max().strftime('%Y-%m-%d')}"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Total Amount", f"${total_amount:,.2f}")
    with col3:
        st.metric("Date Range", date_range)
    
    # Show category breakdown
    st.subheader("Spending by Category")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Category', 'Total Amount', 'Transaction Count']
    st.dataframe(category_totals)
    
    # Show sample transactions
    st.subheader("Sample Transactions")
    st.dataframe(df.head())

def main():
    st.set_page_config(
        page_title="AI Financial Coach with Google ADK + Ollama",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with API key info and CSV template
    with st.sidebar:
        st.title("🔑 Setup & Templates")
        st.info("Local LLM: Ollama must be running at http://localhost:11434 with Qwen installed.")
        st.caption("This application uses Google ADK and a local Ollama model for financial analysis.")
        
        st.divider()
        
        # Add CSV template download
        st.subheader("📊 CSV Template")
        st.markdown("""
        Download the template CSV file with the required format:
        - Date (YYYY-MM-DD)
        - Category
        - Amount (numeric)
        """)
        
        # Create sample CSV content
        sample_csv = """Date,Category,Amount
2024-01-01,Housing,1200.00
2024-01-02,Food,150.50
2024-01-03,Transportation,45.00"""
        
        st.download_button(
            label="📥 Download CSV Template",
            data=sample_csv,
            file_name="expense_template.csv",
            mime="text/csv"
        )
    
    # The offline audit works even without an API key.
    # Main content
    st.title("📊 AI Financial Coach with Google ADK + Ollama")
    st.caption("Google ADK + Ollama (live mode) | Rule-based audit test (offline mode)")
    st.info("This tool analyzes your financial data and provides tailored recommendations for budgeting, savings, and debt management using multiple specialized AI agents.")
    st.divider()
    
    # Create tabs for different sections
    input_tab, about_tab = st.tabs(["💼 Financial Information", "ℹ️ About"])
    
    with input_tab:
        st.header("Enter Your Financial Information")
        st.caption("In live mode, financial inputs are processed locally by Ollama. Use test data for this demo.")
        
        # Income and Dependants section in a container
        with st.container():
            st.subheader("💰 Income & Household")
            income_col, dependants_col = st.columns([2, 1])
            with income_col:
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=0.0,
                    step=100.0,
                    value=3000.0,
                    key="income",
                    help="Enter your total monthly income after taxes"
                )
            with dependants_col:
                dependants = st.number_input(
                    "Number of Dependants",
                    min_value=0,
                    step=1,
                    value=0,
                    key="dependants",
                    help="Include all dependants in your household"
                )
        
        st.divider()
        
        # Expenses section
        with st.container():
            st.subheader("💳 Expenses")
            expense_option = st.radio(
                "How would you like to enter your expenses?",
                ("📤 Upload CSV Transactions", "✍️ Enter Manually"),
                key="expense_option",
                horizontal=True
            )
            
            transaction_file = None
            manual_expenses = {}
            use_manual_expenses = False
            transactions_df = None

            if expense_option == "📤 Upload CSV Transactions":
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("""
                    #### Upload your transaction data
                    Your CSV file should have these columns:
                    - 📅 Date (YYYY-MM-DD)
                    - 📝 Category
                    - 💲 Amount
                    """)
                    
                    transaction_file = st.file_uploader(
                        "Choose your CSV file",
                        type=["csv"],
                        key="transaction_file",
                        help="Upload a CSV file containing your transactions"
                    )
                
                if transaction_file is not None:
                    # Validate CSV format
                    is_valid, message = validate_csv_format(transaction_file)
                    
                    if is_valid:
                        try:
                            # Parse CSV content
                            transaction_file.seek(0)
                            file_content = transaction_file.read()
                            parsed_data = parse_csv_transactions(file_content)
                            
                            # Create DataFrame
                            transactions_df = pd.DataFrame(parsed_data['transactions'])
                            
                            # Display preview
                            display_csv_preview(transactions_df)
                            
                            st.success("✅ Transaction file uploaded and validated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error processing CSV file: {str(e)}")
                            transactions_df = None
                    else:
                        st.error(message)
                        transactions_df = None
            else:
                use_manual_expenses = True
                st.markdown("#### Enter your monthly expenses by category")
                
                # Define expense categories with emojis
                categories = [
                    ("🏠 Housing", "Housing"),
                    ("🔌 Utilities", "Utilities"),
                    ("🍽️ Food", "Food"),
                    ("🚗 Transportation", "Transportation"),
                    ("🏥 Healthcare", "Healthcare"),
                    ("🎭 Entertainment", "Entertainment"),
                    ("👤 Personal", "Personal"),
                    ("💰 Savings", "Savings"),
                    ("📦 Other", "Other")
                ]
                
                # Create three columns for better layout
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                
                # Distribute categories across columns
                for i, (emoji_cat, cat) in enumerate(categories):
                    with cols[i % 3]:
                        manual_expenses[cat] = st.number_input(
                            emoji_cat,
                            min_value=0.0,
                            step=50.0,
                            value=0.0,
                            key=f"manual_{cat}",
                            help=f"Enter your monthly {cat.lower()} expenses"
                        )
                
                if manual_expenses and any(manual_expenses.values()):
                    st.markdown("#### 📊 Summary of Entered Expenses")
                    manual_df_disp = pd.DataFrame({
                        'Category': list(manual_expenses.keys()),
                        'Amount': list(manual_expenses.values())
                    })
                    manual_df_disp = manual_df_disp[manual_df_disp['Amount'] > 0]
                    if not manual_df_disp.empty:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(
                                manual_df_disp,
                                column_config={
                                    "Category": "Category",
                                    "Amount": st.column_config.NumberColumn(
                                        "Amount",
                                        format="$%.2f"
                                    )
                                },
                                hide_index=True
                            )
                        with col2:
                            st.metric(
                                "Total Monthly Expenses",
                                f"${manual_df_disp['Amount'].sum():,.2f}"
                            )
        
        st.divider()
        
        # Debt Information section
        with st.container():
            st.subheader("🏦 Debt Information")
            st.info("Enter your debts to get personalized payoff strategies using both avalanche and snowball methods.")
            
            num_debts = st.number_input(
                "How many debts do you have?",
                min_value=0,
                max_value=10,
                step=1,
                value=0,
                key="num_debts"
            )
            
            debts = []
            if num_debts > 0:
                # Create columns for debts
                cols = st.columns(min(num_debts, 3))  # Max 3 columns per row
                for i in range(num_debts):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"##### Debt #{i+1}")
                        debt_name = st.text_input(
                            "Name",
                            value=f"Debt {i+1}",
                            key=f"debt_name_{i}",
                            help="Enter a name for this debt (e.g., Credit Card, Student Loan)"
                        )
                        debt_amount = st.number_input(
                            "Amount ($)",
                            min_value=0.01,
                            step=100.0,
                            value=1000.0,
                            key=f"debt_amount_{i}",
                            help="Enter the current balance of this debt"
                        )
                        interest_rate = st.number_input(
                            "Interest Rate (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            value=5.0,
                            key=f"debt_rate_{i}",
                            help="Enter the annual interest rate"
                        )
                        min_payment = st.number_input(
                            "Minimum Payment ($)",
                            min_value=0.0,
                            step=10.0,
                            value=50.0,
                            key=f"debt_min_payment_{i}",
                            help="Enter the minimum monthly payment required"
                        )
                        
                        debts.append({
                            "name": debt_name,
                            "amount": debt_amount,
                            "interest_rate": interest_rate,
                            "min_payment": min_payment
                        })
                        
                        if col_idx == 2 or i == num_debts - 1:  # Add spacing after every 3 debts or last debt
                            st.markdown("---")
        
        st.divider()
        
        # Keep the audit test usable when a free LLM provider is rate-limited.
        execution_mode = st.radio(
            "Execution mode",
            ["Full dashboard demo (synthetic results)", "Offline validation demo (no LLM)", "Live AI agents (Ollama)"],
            key="execution_mode",
            help="Offline mode tests the new Python guardrail using clearly marked sample agent outputs."
        )
        if execution_mode.startswith("Offline") or execution_mode.startswith("Full"):
            demo_case = st.selectbox(
                "Offline test case",
                ["Unrealistic savings recommendation", "Incorrect AI expense total", "Consistent agent outputs"],
                key="demo_case"
            )
            st.caption("Offline examples are synthetic fixtures, NOT responses from the four agents.")

        # Analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔄 Analyze My Finances",
                key="analyze_button",
                use_container_width=True,
                help="Click to get your personalized financial analysis"
            )
        
        if analyze_button:
            if expense_option == "📤 Upload CSV Transactions" and transactions_df is None:
                st.error("Please upload a valid transaction CSV file or choose manual entry.")
                return
            if use_manual_expenses and (not manual_expenses or not any(manual_expenses.values())):
                st.warning("No manual expenses entered. Analysis might be limited.")

            st.header("Financial Analysis Results")
            with st.spinner("🤖 AI agents are analyzing your financial data..."): 
                financial_data = {
                    "monthly_income": monthly_income,
                    "dependants": dependants,
                    "transactions": transactions_df.to_dict('records') if transactions_df is not None else None,
                    "manual_expenses": manual_expenses if use_manual_expenses else None,
                    "debts": debts
                }
                
                if execution_mode.startswith("Full"):
                    # Full dashboard integration test with clearly labeled synthetic outputs.
                    # The original four LlmAgent definitions are unchanged and not called.
                    results = create_demo_results(financial_data, demo_case)
                    validation = validate_financial_feasibility(financial_data, results)
                    st.info("FULL DASHBOARD TEST — Synthetic results, no LLM calls. The four real agents remain available in Live mode.")
                    st.caption("All strategies and debt payoff numbers shown below are test examples, not generated financial advice.")
                    display_financial_audit(validation, results["financial_audit"])
                    tabs = st.tabs(["💰 Budget Analysis (test)", "📈 Savings Strategy (test)", "💳 Debt Reduction (test)"])
                    with tabs[0]:
                        display_budget_analysis(results["budget_analysis"])
                    with tabs[1]:
                        display_savings_strategy(results["savings_strategy"])
                    with tabs[2]:
                        display_debt_reduction(results["debt_reduction"])
                    with st.expander("View complete synthetic test outputs"):
                        st.json(results)
                    return

                if execution_mode.startswith("Offline"):
                    # Example agent results for testing the validation rules, not AI responses.
                    expenses = (sum(float(t["Amount"]) for t in (financial_data["transactions"] or []))
                                if financial_data["transactions"] else
                                sum(float(x) for x in (financial_data["manual_expenses"] or {}).values()))
                    debts_total = sum(float(d.get("amount", 0)) for d in debts)
                    cash_flow = monthly_income - expenses - sum(float(d.get("min_payment", 0)) for d in debts)
                    savings_amount = max(0.0, cash_flow * 0.25)
                    reported_expenses = expenses
                    if demo_case == "Unrealistic savings recommendation":
                        savings_amount = max(100.0, cash_flow + 200.0)
                    elif demo_case == "Incorrect AI expense total":
                        reported_expenses = expenses + 100.0
                    demo_results = {
                        "budget_analysis": {"total_expenses": reported_expenses},
                        "savings_strategy": {"recommendations": [{"category": "Example allocation", "amount": savings_amount}]},
                        "debt_reduction": {"total_debt": debts_total}
                    }
                    st.info("OFFLINE TEST: These are synthetic agent outputs for testing the new validation layer. No LLM is called.")
                    st.caption("The four-agent Google ADK workflow is preserved in the live mode, but is not executed here.")
                    validation = validate_financial_feasibility(financial_data, demo_results)
                    display_financial_audit(validation, None)
                    with st.expander("View test inputs / sample agent outputs"):
                        st.json(demo_results)
                    return


                finance_system = FinanceAdvisorSystem()

                try:
                    results = asyncio.run(finance_system.analyze_finances(financial_data))

                    # Check the original numbers independently from the LLM
                    validation = validate_financial_feasibility(financial_data, results)
                    display_financial_audit(validation, results.get("financial_audit"))
                    
                    tabs = st.tabs(["💰 Budget Analysis", "📈 Savings Strategy", "💳 Debt Reduction"])
                    
                    with tabs[0]:
                        st.subheader("Budget Analysis")
                        if "budget_analysis" in results and results["budget_analysis"]:
                            display_budget_analysis(results["budget_analysis"])
                        else:
                            st.write("No budget analysis available.")
                    
                    with tabs[1]:
                        st.subheader("Savings Strategy")
                        if "savings_strategy" in results and results["savings_strategy"]:
                            display_savings_strategy(results["savings_strategy"])
                        else:
                            st.write("No savings strategy available.")
                    
                    with tabs[2]:
                        st.subheader("Debt Reduction Plan")
                        if "debt_reduction" in results and results["debt_reduction"]:
                            display_debt_reduction(results["debt_reduction"])
                        else:
                            st.write("No debt reduction plan available.")
                except Exception as e:
                    logger.exception("AI analysis unavailable")
                    st.warning("AI analysis could not be completed. The independent financial checks are still available below.")
                    st.caption(f"Provider error: {str(e)[:250]}")
                    # Run only the Python checks; no fake AI recommendations
                    validation = validate_financial_feasibility(financial_data, {})
                    display_financial_audit(validation, None)
    
    with about_tab:
        st.markdown("""
        ### About AI Financial Coach
        
        This application uses Google ADK with Ollama to run four specialized agents:
        
        1. **🔍 Budget Analysis Agent**
           - Analyzes spending patterns
           - Identifies areas for cost reduction
           - Provides actionable recommendations
        
        2. **💰 Savings Strategy Agent**
           - Creates personalized savings plans
           - Calculates emergency fund requirements
           - Suggests automation techniques
        
        3. **💳 Debt Reduction Agent**
           - Develops optimal debt payoff strategies
           - Compares different repayment methods
           - Provides actionable debt reduction tips
        
        4. **✅ Financial Audit Agent**
           - Reviews recommendations for inconsistencies
           - Complements Python-based financial checks

        ### Privacy & Security
        
        - Live-mode financial inputs are processed by your locally running Ollama model
        - Sessions are held in memory while the application runs
        - Do not enter real sensitive information in a demo deployment
        
        ### Need Help?
        
        For support or questions:
        - Check the [documentation](https://github.com/Shubhamsaboo/awesome-llm-apps)
        - Report issues on [GitHub](https://github.com/Shubhamsaboo/awesome-llm-apps/issues)
        """)

if __name__ == "__main__":
    main()