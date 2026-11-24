"""Check whether Ollama returns a valid BudgetAnalysis payload."""
import os
import requests
from dotenv import load_dotenv
from ai_financial_coach_agent import BudgetAnalysis

load_dotenv()
model = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
base_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')

schema = BudgetAnalysis.model_json_schema()
expenses = {'Housing': 1900, 'Food': 900, 'Transportation': 450, 'Utilities': 350, 'Healthcare': 300}
message = (
    'You are the BudgetAnalysisAgent. Analyze the following synthetic monthly finances. '
    'Monthly income: $3000. Number of dependants: 2. '
    'Monthly expense categories: ' + str(expenses) + '. '
    'Return a valid BudgetAnalysis object that follows the provided JSON schema. '
    'The five spending categories must sum to $3900. '
    'For recommendations, provide 2 actionable suggestions. '
    'All numbers must be JSON numbers, not currency-formatted strings. '
    'Return only JSON, no markdown.'
)

print(f'Testing complete BudgetAnalysis schema with {model} ...', flush=True)
try:
    response = requests.post(
        f'{base_url}/api/chat',
        json={
            'model': model,
            'messages': [{'role': 'user', 'content': message}],
            'format': schema,
            'stream': False,
            'options': {'temperature': 0, 'num_ctx': 4096},
        },
        timeout=240,
    )
    response.raise_for_status()
    content = response.json()['message']['content']
    budget = BudgetAnalysis.model_validate_json(content)
    print('BUDGET SCHEMA TEST: OK')
    print(f'Expenses: ${budget.total_expenses:,.2f}; categories: {len(budget.spending_categories)}')
    print(f'Recommendations: {len(budget.recommendations)}')
except Exception as exc:
    print(f'BUDGET SCHEMA TEST: FAILED ({type(exc).__name__}: {exc})')
    raise SystemExit(1)
