"""
AI Sales Intelligence Agent — Dashboard

A presentable, non-technical front end over agent.py, built to show (not
just tell) how the prototype works: live metric tiles pulled through the
YAML semantic model, and a chat box that routes questions to either the
semantic model or the retrieval/vector-search index over policy docs.

Run with:  streamlit run dashboard.py
"""

import streamlit as st

from agent import SalesIntelligenceAgent

st.set_page_config(page_title="AI Sales Intelligence Agent", page_icon="\U0001F4CA", layout="wide")

NAVY = "#1F3864"
SLATE = "#44546A"
BG_CARD = "#F4F6FA"
GREEN = "#2E7D32"
BORDER = "#D9DEE8"

st.markdown(
    f"""
    <style>
    .stat-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 18px 20px;
        text-align: left;
    }}
    .stat-label {{
        font-size: 13px;
        color: {SLATE};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .stat-value {{
        font-size: 28px;
        font-weight: 700;
        color: {NAVY};
    }}
    .mode-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        color: white;
    }}
    .source-tag {{
        display: inline-block;
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        color: {SLATE};
        margin-right: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_agent():
    return SalesIntelligenceAgent()


agent = load_agent()

# ---- header ----------------------------------------------------------
col_title, col_mode = st.columns([4, 1])
with col_title:
    st.markdown(f"<h1 style='color:{NAVY};margin-bottom:0;'>AI Sales Intelligence Agent</h1>", unsafe_allow_html=True)
    st.caption("Prototype — grounded in a YAML semantic model + a retrieval index over sales policy docs")
with col_mode:
    mode_color = GREEN if agent.live_mode else "#8A6D00"
    mode_text = "LIVE — OpenAI" if agent.live_mode else "OFFLINE DEMO"
    st.markdown(
        f"<div style='text-align:right;padding-top:18px;'>"
        f"<span class='mode-badge' style='background:{mode_color};'>{mode_text}</span></div>",
        unsafe_allow_html=True,
    )

st.write("")

# ---- stat tiles --------------------------------------------------------
pipeline = agent.semantic_model.query_metric("pipeline")
bookings = agent.semantic_model.query_metric("bookings")
win_rate = agent.semantic_model.query_metric("win_rate")
avg_disc = agent.semantic_model.query_metric("avg_discount")

c1, c2, c3, c4 = st.columns(4)
tiles = [
    (c1, "Open Pipeline", f"${pipeline['value']:,.0f}"),
    (c2, "Bookings (Closed Won)", f"${bookings['value']:,.0f}"),
    (c3, "Win Rate", f"{win_rate['value'] * 100:.0f}%"),
    (c4, "Avg. Discount (Closed Won)", f"{avg_disc['value']:.1f}%"),
]
for col, label, value in tiles:
    with col:
        st.markdown(
            f"<div class='stat-card'><div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )

st.write("")
left, right = st.columns([3, 2])

# ---- chat -------------------------------------------------------------
with left:
    st.subheader("Ask the agent")
    st.caption("Try: \"What's our pipeline in AMER?\" or \"What discount needs VP approval?\"")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Question", label_visibility="collapsed", placeholder="Ask a sales question…")
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Routing to the right tool…"):
            answer = agent.answer(question.strip())
        st.session_state.history.insert(0, (question.strip(), answer))

    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Agent:** {a}")
        st.markdown("---")

# ---- how it works -------------------------------------------------------
with right:
    st.subheader("How it's grounded")
    tab1, tab2 = st.tabs(["Semantic model (YAML)", "Knowledge base"])
    with tab1:
        st.caption("Metrics are defined once here, not hardcoded per question.")
        with open("semantic_model.yaml") as f:
            st.code(f.read(), language="yaml")
    with tab2:
        st.caption("Retrieved via a vector search index, not hardcoded per question.")
        import glob
        for path in sorted(glob.glob("knowledge_base/*.md")):
            with open(path) as f:
                with st.expander(path.split("/")[-1]):
                    st.markdown(f.read())
