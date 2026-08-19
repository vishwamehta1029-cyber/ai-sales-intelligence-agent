"""
Prototype AI sales-intelligence agent.

Built as a rapid personal project to demonstrate, hands-on, the pattern
described in F5's AI Sales Intelligence Analyst posting: a conversational
agent that grounds answers in (1) a governed YAML semantic model over
sales/pipeline data and (2) a retrieval/vector search index over
policy & playbook docs, routing each question to the right tool.

Two run modes:
  - LIVE mode: if OPENAI_API_KEY is set, an actual LLM (via OpenAI's
    function-calling / tool-use API) decides which tool to call and
    writes the final natural-language answer.
  - OFFLINE DEMO mode: if no key is present, a small keyword router calls
    the exact same tool functions, so the semantic-model and
    retrieval logic can be verified end-to-end without network access.
    This is clearly a fallback for demoing offline, not a claim of full
    LLM orchestration.
"""

import json
import os
import re

import pandas as pd
import yaml

from vector_search import VectorSearchIndex

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_semantic_model",
            "description": (
                "Query a governed sales metric (pipeline, bookings, win_rate, "
                "avg_discount) defined in the YAML semantic model, optionally "
                "filtered to one region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "enum": ["pipeline", "bookings", "win_rate", "avg_discount"],
                    },
                    "region": {
                        "type": "string",
                        "enum": ["AMER", "EMEA", "APAC"],
                        "description": "Optional region filter. Omit for company-wide.",
                    },
                },
                "required": ["metric_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Retrieve relevant snippets from sales policy/playbook docs "
                "(discount approval rules, territory definitions, pipeline "
                "review triggers) via the vector search index."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a sales intelligence assistant for a B2B sales org. Answer "
    "only using the provided tools — never invent numbers or policy "
    "details. Use query_semantic_model for pipeline/bookings/win-rate/"
    "discount questions, and search_docs for policy or playbook questions. "
    "Keep answers to 1-3 sentences."
)


class SemanticModel:
    """Loads semantic_model.yaml and answers metric queries against sales_data.csv."""

    def __init__(self, yaml_path="semantic_model.yaml"):
        with open(yaml_path) as f:
            self.spec = yaml.safe_load(f)
        self.df = pd.read_csv(self.spec["source"]["file"])
        # resolve logical field names (e.g. "amount") to actual columns (e.g. "amount_usd")
        self.field_to_column = {
            name: attrs["column"] for name, attrs in self.spec["entities"]["opportunity"]["fields"].items()
        }

    def _apply_filter(self, df, filter_expr):
        if not filter_expr:
            return df
        # tiny, safe-ish expression translator for this demo's filter grammar
        expr = filter_expr
        expr = expr.replace("stage not in ['Closed Won', 'Closed Lost']",
                             "~df['stage'].isin(['Closed Won', 'Closed Lost'])")
        expr = expr.replace("stage == 'Closed Won'", "df['stage'] == 'Closed Won'")
        expr = expr.replace("stage in ['Closed Won', 'Closed Lost']",
                             "df['stage'].isin(['Closed Won', 'Closed Lost'])")
        mask = eval(expr, {"df": df})
        return df[mask]

    def query_metric(self, metric_name, region=None):
        metric = self.spec["metrics"].get(metric_name)
        if not metric:
            return f"Unknown metric '{metric_name}'. Known metrics: {list(self.spec['metrics'])}"

        df = self.df
        if region:
            df = df[df["region"].str.upper() == region.upper()]

        if metric["agg"] == "ratio":
            num = len(self._apply_filter(df, metric["numerator_filter"]))
            den = len(self._apply_filter(df, metric["denominator_filter"]))
            value = round(num / den, 3) if den else None
            return {"metric": metric_name, "region": region or "ALL", "value": value, "description": metric["description"]}

        filtered = self._apply_filter(df, metric.get("filter"))
        col = self.field_to_column.get(metric.get("field"), metric.get("field"))
        if metric["agg"] == "sum":
            value = float(filtered[col].sum())
        elif metric["agg"] == "mean":
            value = round(float(filtered[col].mean()), 2) if len(filtered) else None
        else:
            value = None

        return {"metric": metric_name, "region": region or "ALL", "value": value, "description": metric["description"]}


class SalesIntelligenceAgent:
    REGIONS = ["AMER", "EMEA", "APAC"]

    def __init__(self):
        self.semantic_model = SemanticModel()
        self.doc_index = VectorSearchIndex()
        self.live_mode = bool(os.environ.get("OPENAI_API_KEY"))
        self._client = None
        if self.live_mode:
            from openai import OpenAI
            self._client = OpenAI()  # reads OPENAI_API_KEY from env

    # ---- tools -----------------------------------------------------
    def tool_query_semantic_model(self, metric_name, region=None):
        return self.semantic_model.query_metric(metric_name, region)

    def tool_search_docs(self, query):
        return self.doc_index.search(query)

    def _dispatch_tool(self, name, args):
        if name == "query_semantic_model":
            return self.tool_query_semantic_model(args.get("metric_name"), args.get("region"))
        if name == "search_docs":
            return self.tool_search_docs(args.get("query", ""))
        return {"error": f"unknown tool {name}"}

    # ---- LIVE mode: real LLM tool-calling ---------------------------
    def answer_live(self, question, max_hops=3):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for _ in range(max_hops):
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content

            messages.append(msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = self._dispatch_tool(call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                })
        return "Reached max tool-call hops without a final answer."

    # ---- OFFLINE DEMO mode: keyword router, same tools --------------
    def _extract_region(self, text):
        for r in self.REGIONS:
            if r.lower() in text.lower():
                return r
        return None

    def answer(self, question):
        if self.live_mode:
            return self.answer_live(question)
        return self.answer_offline(question)

    def answer_offline(self, question):
        q = question.lower()
        region = self._extract_region(question)

        metric_keywords = {
            "pipeline": "pipeline",
            "bookings": "bookings",
            "win rate": "win_rate",
            "discount": "avg_discount",
        }

        for kw, metric in metric_keywords.items():
            if kw in q and "polic" not in q and "approv" not in q:
                result = self.tool_query_semantic_model(metric, region)
                return self._format_metric_answer(result)

        # otherwise treat as a policy / playbook question -> retrieval
        hits = self.tool_search_docs(question)
        if not hits:
            return "I couldn't find anything relevant in the semantic model or the knowledge base."
        lines = [f"- ({h['source']}) {h['text']}" for h in hits]
        return "Based on the sales playbook and policy docs:\n" + "\n".join(lines)

    def _format_metric_answer(self, result):
        if isinstance(result, str):
            return result
        val = result["value"]
        if result["metric"] in ("pipeline", "bookings"):
            val_str = f"${val:,.0f}"
        elif result["metric"] == "win_rate":
            val_str = f"{val * 100:.1f}%" if val is not None else "n/a"
        elif result["metric"] == "avg_discount":
            val_str = f"{val:.1f}%" if val is not None else "n/a"
        else:
            val_str = str(val)
        return f"{result['metric']} ({result['region']}): {val_str} — {result['description']}"


if __name__ == "__main__":
    agent = SalesIntelligenceAgent()
    mode = "LIVE (OpenAI function-calling)" if agent.live_mode else "OFFLINE DEMO (keyword router, same tools)"
    print(f"Run mode: {mode}\n")

    questions = [
        "What's our open pipeline in EMEA?",
        "What's our overall win rate?",
        "What's our average discount on closed-won deals?",
        "What discount level needs VP approval?",
        "How are territories defined for APAC?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {agent.answer(q)}\n")
