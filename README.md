# AI Financial Coach — Financial Audit Extension

This project extends the original AI Financial Coach from [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps) with:

- A fourth Google ADK `LlmAgent` named `FinancialAuditAgent` in the sequential workflow.
- A separate Python financial validation layer that checks budget totals, savings affordability and debt balances.
- A Streamlit audit panel showing rule-based checks and the AI audit separately.
- Unit tests for five financial scenarios.

## Installation

Keep the original `requirements.txt` from the repository. Copy `.env.example` to `.env`, then add your **own** Google API key. Do not commit `.env`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run ai_financial_coach_agent.py
```

Run the checks independently of the model API:

```powershell
.\.venv\Scripts\python.exe -m unittest test_financial_validator.py -v
```

## Scope and limitations

The new deterministic layer checks affordability based on monthly income minus monthly expenses and minimum debt payments. It assumes savings recommendations are **additional monthly allocations**. It does not replace professional financial advice or a complete loan amortization engine. The LLM-based auditor provides a qualitative review; its correctness is not guaranteed. Inputs are sent to the Google Gemini API, so avoid sensitive data in demonstrations.

A fourth agent and five tests are architectural additions, **not** evidence of a measured 10% performance gain. Run a controlled baseline-versus-improved evaluation before claiming a percentage.

## Source attribution

Adapted from the original [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps). Retain the original license and attribution notices from the repository when publishing.
