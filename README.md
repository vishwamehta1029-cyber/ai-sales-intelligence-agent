# AI Sales Intelligence Agent
### Ask a sales question. Get an answer grounded in metrics and policy.

A conversational analytics prototype that combines governed CRM metrics with document retrieval, keeping calculations in deterministic tools and using an LLM to select tools and explain results.

## The business problem
Pipeline questions and policy questions need different evidence. A numerical answer should come from a defined metric; a discount rule should come from a policy document. This project gives both a clear source.

## What I built
- **Semantic model:** YAML definitions for pipeline, bookings, win rate, and average discount over sample CRM data.
- **Policy retrieval:** TF-IDF and cosine-similarity search over discount and territory documents.
- **Tool layer:** separate metric-query and document-search functions.
- **Agent:** OpenAI function calling in live mode; a keyword router for an offline demonstration.
- **Interface:** Streamlit dashboard exposing the metrics, chat, and grounding material.

**Stack:** Python · Streamlit · pandas · OpenAI API · scikit-learn · YAML

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Or run `python agent.py` for the command-line demonstration.

Without `OPENAI_API_KEY`, the app uses the offline router. Configure that environment variable to use live OpenAI tool calling. Keep the key out of source control.

## Try asking
- What's our open pipeline in EMEA?
- What's our overall win rate?
- What's our average discount on closed-won deals?
- What discount level needs VP approval?
- How are territories defined for APAC?

These questions use the repository's sample CRM records and policy documents, not a live Salesforce connection.

## Architecture decisions
| Decision | Why |
| --- | --- |
| Metrics defined in YAML | Keep business definitions separate from natural-language phrasing. |
| Deterministic calculations | Let code compute the number rather than asking an LLM to invent it. |
| Shared tools in both modes | Make the data/retrieval workflow explorable without API credentials. |
| Lightweight lexical retrieval | Keep the demo local and inspectable; this is TF-IDF, not neural embeddings. |

## Scope & next steps
Portfolio prototype. It uses mock CRM data and a small local policy corpus. Production work would include warehouse integration, access controls, embedding-based retrieval if useful for the corpus, and evaluation cases for routing, numerical correctness, and answer faithfulness.

---
Built by [Vishwa Mehta](https://www.linkedin.com/in/vishwa10/) · Forward Deployed Engineer
