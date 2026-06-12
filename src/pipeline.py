"""End-to-end pipeline: dataset prep -> embeddings -> topics -> 2D map.

Stages:
1. Load and clean the BBC News dataset (SetFit/bbc-news).
2. Embed every article with all-MiniLM-L6-v2.
3. Fit BERTopic on the precomputed embeddings (fallback: UMAP + KMeans).
4. Project embeddings to 2D with UMAP for the scatter map.
5. Save parquet/CSV artifacts consumed by the Streamlit app.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import config


def load_and_clean() -> pd.DataFrame:
    """Checkpoint 1: load the dataset, clean it, and persist stats."""
    from datasets import load_dataset

    print(f"Loading dataset {config.DATASET_NAME} ...")
    ds = load_dataset(config.DATASET_NAME)

    frames = [split.to_pandas() for split in ds.values()]
    df = pd.concat(frames, ignore_index=True)

    # SetFit/bbc-news ships `text`, integer `label`, and `label_text`.
    # Prefer the readable label when present.
    if "label_text" in df.columns:
        df["label"] = df["label_text"]
    df = df[[c for c in ("text", "label") if c in df.columns]]

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].dropna(subset=["text"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if len(df) > config.MAX_ROWS:
        df = df.sample(n=config.MAX_ROWS, random_state=config.RANDOM_STATE)
        df = df.reset_index(drop=True)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_PARQUET, index=False)

    lines = [
        f"Dataset: {config.DATASET_NAME}",
        f"Rows after cleaning: {len(df)}",
        "",
    ]
    if "label" in df.columns:
        lines.append("Label distribution:")
        for label, count in df["label"].value_counts().items():
            lines.append(f"  {label}: {count}")
    config.DATA_SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    print(f"Saved {config.CLEAN_PARQUET} ({len(df)} rows)")
    return df


def embed_texts(texts: list[str]) -> np.ndarray:
    """Checkpoint 2a: sentence embeddings with MiniLM."""
    from sentence_transformers import SentenceTransformer

    print(f"Embedding {len(texts)} documents with {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.asarray(embeddings)


def fit_bertopic(texts: list[str], embeddings: np.ndarray):
    """Fit BERTopic on precomputed embeddings. Returns (topic_ids, topic_info)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # "leaf" selection yields finer-grained clusters than the default "eom",
    # which otherwise merges business + politics into one giant topic.
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=config.RANDOM_STATE,
    )
    # Strip English stopwords so topic labels are readable keywords,
    # not "the, to, of".
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=config.MIN_TOPIC_SIZE,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    # Cap topic count to keep the map readable.
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    if n_topics > 15:
        print(f"{n_topics} topics found; reducing to 12 ...")
        topic_model.reduce_topics(texts, nr_topics=12)
        topics = topic_model.topics_

    # Assign outlier (-1) docs to their nearest topic so every point is colored.
    if -1 in topics:
        topics = topic_model.reduce_outliers(texts, topics, strategy="embeddings",
                                             embeddings=embeddings)
        # Re-pass the vectorizer, otherwise update_topics falls back to a
        # default CountVectorizer and labels degrade to stopwords.
        topic_model.update_topics(texts, topics=topics,
                                  vectorizer_model=vectorizer_model)

    topic_words = {
        tid: [w for w, _ in topic_model.get_topic(tid)][:5]
        for tid in sorted(set(topics))
    }
    return np.asarray(topics), topic_words


def fit_fallback_kmeans(embeddings: np.ndarray, texts: list[str]):
    """Fallback: UMAP (5D) + KMeans(k=10) + c-TF-IDF-style top words via TF-IDF."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from umap import UMAP

    print("Falling back to UMAP + KMeans(k=10) ...")
    reducer = UMAP(n_components=5, metric="cosine", random_state=config.RANDOM_STATE)
    reduced = reducer.fit_transform(embeddings)
    topics = KMeans(n_clusters=10, random_state=config.RANDOM_STATE, n_init=10).fit_predict(reduced)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    topic_words = {}
    for tid in sorted(set(topics)):
        mask = topics == tid
        mean_scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        topic_words[tid] = vocab[np.argsort(mean_scores)[::-1][:5]].tolist()
    return np.asarray(topics), topic_words


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    """Checkpoint 2c: 2D UMAP projection for the scatter map."""
    from umap import UMAP

    print("Projecting embeddings to 2D with UMAP ...")
    reducer = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=config.RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


def main() -> None:
    df = load_and_clean()
    texts = df["text"].tolist()
    embeddings = embed_texts(texts)

    try:
        topics, topic_words = fit_bertopic(texts, embeddings)
    except Exception as exc:  # noqa: BLE001 — fallback mandated by spec
        print(f"BERTopic failed ({exc!r}); using fallback.", file=sys.stderr)
        topics, topic_words = fit_fallback_kmeans(embeddings, texts)

    coords = project_2d(embeddings)

    df["topic_id"] = topics
    df["topic_label"] = [
        f"{tid}: " + ", ".join(topic_words[tid]) for tid in topics
    ]
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    df.to_parquet(config.EMBEDDED_TOPICS_PARQUET, index=False)
    print(f"Saved {config.EMBEDDED_TOPICS_PARQUET}")

    summary = (
        df.groupby("topic_id")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    summary["top_words"] = summary["topic_id"].map(
        lambda tid: ", ".join(topic_words[tid])
    )
    summary.to_csv(config.TOPIC_SUMMARY_CSV, index=False)
    print(f"Saved {config.TOPIC_SUMMARY_CSV}")

    n_topics = summary.shape[0]
    biggest_share = summary["count"].max() / len(df)
    print(f"Topics: {n_topics}; largest topic share: {biggest_share:.1%}")


if __name__ == "__main__":
    main()
