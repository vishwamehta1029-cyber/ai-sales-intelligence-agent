# AI Sales Intelligence Agent — Prototype

A small, working prototype built to demonstrate the core pattern in F5's
AI Sales Intelligence Analyst role: a conversational agent that answers
sales/pipeline questions in natural language by grounding its answers in
(1) a governed semantic model and (2) a retrieval index over policy docs,
rather than hallucinating numbers or free-text answers.

**What this is:** a personal project, built quickly to show the pattern
and prove I can pick this stack up fast — not a claim of prior production
experience with these specific tools.

## Dashboard

`dashboard.py` is a Streamlit front end over `agent.py` — stat tiles for
pipeline/bookings/win rate/discount, a chat box to ask questions, and a
panel showing the underlying YAML semantic model and knowledge-base docs
so it's visible what's grounding each answer. Run it with:

```
pip install streamlit
streamlit run dashboard.py
```

## Components

- `semantic_model.yaml` — defines business metrics (pipeline, bookings,
  win rate, average discount) as governed aggregations over the
  underlying data, instead of one-off SQL per question. Adding a new
  metric or dimension means editing YAML, not code.
- `sales_data.csv` — mock CRM/opportunity data standing in for a
  Salesforce export.
- `knowledge_base/` — short policy and playbook docs (discounting rules,
  territory definitions, pipeline-review triggers).
- `vector_search.py` — a retrieval/vector search index over those docs
  (TF-IDF + cosine similarity, so it runs fully offline with no API key;
  swapping in real embeddings or a hosted vector DB is a localized
  change — see comments in the file).
- `agent.py` — the agent layer: routes a question to either the
  semantic-model tool or the doc-retrieval tool and returns a grounded
  answer. Supports two modes:
  - **LIVE**: if `OPENAI_API_KEY` is set, a real OpenAI model
    (`gpt-4o-mini`) does the routing via actual function/tool calling
    (implemented with the current OpenAI Python SDK's `tools=` /
    `tool_calls` interface). Verified working end-to-end on a live run —
    see the LIVE sample run below, where the model composes its own
    natural-language answers from tool results rather than reusing any
    hardcoded string.
  - **OFFLINE DEMO**: a keyword router calls the exact same
    `tool_query_semantic_model` / `tool_search_docs` functions the LIVE
    path uses, so the semantic-model and retrieval logic itself is
    fully verified end-to-end without network access (this is the mode
    shown in the sample run below).

## Example run (LIVE mode, verified)

```
Run mode: LIVE (OpenAI function-calling)

Q: What's our open pipeline in EMEA?
A: Our open pipeline in EMEA is $199,000.

Q: What's our overall win rate?
A: Our overall win rate is 70%.

Q: What's our average discount on closed-won deals?
A: Our average discount on closed-won deals is 11.86%.

Q: What discount level needs VP approval?
A: Discounts above 20% require VP of Sales and Finance approval.

Q: How are territories defined for APAC?
A: Territories in APAC are defined as Asia-Pacific and Japan, which are
further divided into Greater China and Rest-of-APAC for reporting
purposes. Also, named accounts with an annual contract value above $5M
are managed by the Strategic Accounts team, irrespective of geography.
```

## Example run (OFFLINE DEMO mode)

```
Run mode: OFFLINE DEMO (keyword router, same tools)

Q: What's our open pipeline in EMEA?
A: pipeline (EMEA): $199,000 — Sum of amount for opportunities not yet closed (open pipeline).

Q: What's our overall win rate?
A: win_rate (ALL): 70.0% — Share of closed opportunities that closed won.

Q: What's our average discount on closed-won deals?
A: avg_discount (ALL): 11.9% — Average discount percentage across closed-won deals.

Q: What discount level needs VP approval?
A: Based on the sales playbook and policy docs:
- (discount_policy.md) Discounts above 20% require VP of Sales and Finance approval...

Q: How are territories defined for APAC?
A: Based on the sales playbook and policy docs:
- (territory_definitions.md) APAC: Asia-Pacific and Japan, split into Greater China and Rest-of-APAC...
```

## What I'd change for a real, production version

- Back the semantic model with Snowflake/BigQuery instead of a CSV.
- Replace the TF-IDF index with real embeddings and a hosted vector
  store (or Snowflake Cortex Search).
- Replace the keyword router with an actual agent framework (ADK,
  LangGraph, or LangChain) doing LLM-driven tool selection in LIVE mode
  by default.
- Add evaluation/regression test cases for routing accuracy and answer
  faithfulness, per the JD's "quality and governance" section.

## Run it yourself

```
pip install pandas scikit-learn pyyaml
python agent.py
```
