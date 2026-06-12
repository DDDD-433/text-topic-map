"""Interactive topic map dashboard for BBC News articles.

Reads only the precomputed artifacts produced by `python -m src.pipeline`:
- data/processed/embedded_topics.parquet
- data/processed/topic_summary.csv
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDED_TOPICS_PARQUET = PROJECT_ROOT / "data" / "processed" / "embedded_topics.parquet"
TOPIC_SUMMARY_CSV = PROJECT_ROOT / "data" / "processed" / "topic_summary.csv"

MAX_PLOT_POINTS = 3000
HOVER_PREVIEW_CHARS = 180

st.set_page_config(page_title="Text Topic Map", page_icon="🗺️", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_parquet(EMBEDDED_TOPICS_PARQUET)
    summary = pd.read_csv(TOPIC_SUMMARY_CSV)
    df["preview"] = df["text"].str.slice(0, HOVER_PREVIEW_CHARS) + "…"
    return df, summary


if not EMBEDDED_TOPICS_PARQUET.exists():
    st.error(
        "Precomputed data not found. Run `bash scripts/run_pipeline.sh` first to "
        "generate `data/processed/embedded_topics.parquet`."
    )
    st.stop()

df, topic_summary = load_data()

# ---------------------------------------------------------------- header
st.title("🗺️ Text Topic Map — BBC News")
st.caption(
    f"{len(df):,} BBC News articles embedded with MiniLM, clustered with BERTopic, "
    "and projected to 2D with UMAP. Each point is one article."
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

topic_options = sorted(df["topic_label"].unique(), key=lambda s: int(s.split(":")[0]))
selected_topics = st.sidebar.multiselect(
    "Topics", topic_options, default=topic_options
)

if "label" in df.columns:
    label_options = sorted(df["label"].unique())
    selected_labels = st.sidebar.multiselect(
        "BBC section label", label_options, default=label_options
    )
else:
    selected_labels = None

search_query = st.sidebar.text_input("Search article text", placeholder="e.g. election, champions league…")

# ---------------------------------------------------------------- filtering
filtered = df[df["topic_label"].isin(selected_topics)]
if selected_labels is not None:
    filtered = filtered[filtered["label"].isin(selected_labels)]
if search_query.strip():
    filtered = filtered[filtered["text"].str.contains(search_query.strip(), case=False, regex=False)]

# ---------------------------------------------------------------- KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Total articles", f"{len(df):,}")
k2.metric("Topics", df["topic_id"].nunique())
k3.metric("Articles matching filters", f"{len(filtered):,}")

# ---------------------------------------------------------------- scatter map
if filtered.empty:
    st.warning("No articles match the current filters. Broaden your topic/label selection or clear the search box.")
else:
    plot_df = filtered
    if len(plot_df) > MAX_PLOT_POINTS:
        plot_df = plot_df.sample(n=MAX_PLOT_POINTS, random_state=0)
        st.caption(f"Showing a random sample of {MAX_PLOT_POINTS:,} points for speed.")

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="topic_label",
        hover_data={"preview": True, "x": False, "y": False, "topic_label": True},
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=620,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(
        legend_title_text="Topic",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(245,246,250,1)",
    )
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- topic panel
st.subheader("Topic summary")
left, right = st.columns([1, 1])

with left:
    display_summary = topic_summary.copy()
    display_summary["share"] = (display_summary["count"] / len(df) * 100).round(1).astype(str) + "%"
    st.dataframe(display_summary, width="stretch", hide_index=True)

with right:
    inspect_topic = st.selectbox("Inspect a topic", topic_options)
    examples = df[df["topic_label"] == inspect_topic].head(5)
    st.markdown(f"**Top 5 example articles — {inspect_topic}**")
    for _, row in examples.iterrows():
        label_note = f" · _{row['label']}_" if "label" in df.columns else ""
        st.markdown(f"- {row['preview']}{label_note}")
