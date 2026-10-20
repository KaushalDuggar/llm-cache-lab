"""LLM Cache Lab: ask questions, then watch hit/miss stats.

    uv run streamlit run dashboard.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from cachelab import embeddings, exact_cache, llm, metrics, pipeline, redis_client, semantic_cache

st.set_page_config(page_title="LLM Cache Lab", layout="wide")

# Fixed color per answer source, so a color always means the same route.
# Light/dark steps of the same hues; text stays in theme ink, never these colors.
SOURCES = ["llm", "exact_cache", "semantic_cache"]
SOURCE_LABELS = {"llm": "LLM (miss)", "exact_cache": "Exact hit", "semantic_cache": "Semantic hit"}
PALETTE = {
    "light": ["#eb6834", "#2a78d6", "#1baf7a"],
    "dark": ["#d95926", "#3987e5", "#199e70"],
}
theme = getattr(getattr(st.context, "theme", None), "type", None) or "light"
colors = PALETTE.get(theme, PALETTE["light"])
source_color = alt.Color(
    "source_label:N",
    title="Answer source",
    scale=alt.Scale(domain=[SOURCE_LABELS[s] for s in SOURCES], range=colors),
)

# Repeats (exact hits), paraphrases (semantic hits) and traps (look similar, mean something else).
SAMPLE_QUESTIONS = """How do I reset my password?
How do I reset my password?
how do i reset my password
I forgot how to change my login password.
how can I recover my account password
What is Redis?
Can you explain what Redis is?
How do I reset my router?
How do I enable 2FA?
How do I disable 2FA?"""


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    models = llm.list_chat_models()
    if not models:
        st.error("No chat models installed.\n\n`ollama pull llama3.2:3b`")
        st.stop()
    default_idx = models.index(llm.DEFAULT_CHAT_MODEL) if llm.DEFAULT_CHAT_MODEL in models else 0
    model = st.selectbox("Model", models, index=default_idx)
    mode_help = {
        "none": "none: always call the LLM",
        "exact": "exact: same text only",
        "semantic": "semantic: same meaning",
        "hybrid": "hybrid: exact → semantic → LLM",
    }
    cache_mode = st.radio("Cache mode", pipeline.CACHE_MODES, index=pipeline.CACHE_MODES.index("hybrid"),
                          format_func=mode_help.get)
    uses_semantic = cache_mode in ("semantic", "hybrid")
    threshold = st.slider("Similarity threshold", 0.50, 0.99, semantic_cache.DEFAULT_THRESHOLD, 0.01,
                          disabled=not uses_semantic,
                          help="A stored question counts as a match only if its cosine similarity is at "
                               "least this. Lower = more hits but more wrong answers.")
    ttl = st.number_input("Cache TTL (seconds)", min_value=10, value=exact_cache.DEFAULT_TTL, step=60,
                          disabled=cache_mode == "none",
                          help="How long a cached answer lives before Redis deletes it (EXPIRE).")
    st.divider()
    if st.button("Reset metrics", help="Clears the request log and the Redis counters (cache:stats)."):
        metrics.reset()
        redis_client.reset_stats()
        st.session_state.pop("last", None)
        st.rerun()
    if st.button("Clear exact cache", help="Deletes every cache:exact:* key."):
        st.toast(f"Deleted {exact_cache.clear()} exact entries")
    if st.button("Clear semantic cache", help="Deletes every cache:semantic:* key (the index stays)."):
        st.toast(f"Deleted {semantic_cache.clear()} semantic entries")


def run(question: str) -> None:
    st.session_state["last"] = pipeline.ask(question, model=model, cache_mode=cache_mode,
                                            ttl=int(ttl), threshold=threshold)


st.title("LLM Cache Lab")
ask_tab, dash_tab = st.tabs(["Ask", "Dashboard"])

# ---------------- ask ----------------
with ask_tab:
    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Question", placeholder="How do I reset my password?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        with st.spinner(f"Asking {model}..."):
            run(question)

    if last := st.session_state.get("last"):
        hit = last["cache_hit"]
        st.markdown(f"**{'✅ Cache HIT' if hit else '❌ Cache MISS'}** · source: `{last['source']}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency", f"{last['latency_ms']:,.0f} ms")
        if hit:
            c2.metric("Model load", "skipped")
            c3.metric("Tokens saved", last["output_tokens"])
            if last["source"] == "semantic_cache":
                c4.metric("Similarity", f"{last['similarity']:.3f}", help=f"threshold {last['threshold']:.2f}")
            else:
                c4.metric("TTL left", f"{last['ttl_left']} s")
        else:
            c2.metric("Model load (cold start)", f"{last['load_ms']:,.0f} ms")
            c3.metric("Output tokens", last["output_tokens"])
            c4.metric("Speed", f"{last['tokens_per_sec']:.1f} tok/s")
        if last.get("cache_key"):
            what = "Found under" if hit else "Stored under"
            st.caption(f"{what} Redis key `{last['cache_key']}`  ·  normalized text: "
                       f"`{exact_cache.normalize(last['question'])}`")
        if last.get("embed_ms") is not None:
            st.caption(f"Semantic lookup: embed {last['embed_ms']:.0f} ms + vector search "
                       f"{last['search_ms']:.0f} ms")
        if hit and last["matched_question"] != last["question"]:
            st.caption(f"Answer was originally generated for: “{last['matched_question']}”")
        if not hit and last.get("nearest_question"):
            st.caption(f"Closest stored question: “{last['nearest_question']}” at similarity "
                       f"{last['similarity']:.3f}, below the {last['threshold']:.2f} threshold, so it was a miss.")
        st.markdown(last["answer"])

    with st.expander("Similarity playground: how close are two sentences?"):
        st.caption("Embeds both with nomic-embed-text and compares the vectors. No LLM, no cache.")
        a = st.text_input("Sentence A", "How do I reset my password?")
        b = st.text_input("Sentence B", "I forgot how to change my login password.")
        if st.button("Compare"):
            (va, _), (vb, _) = embeddings.embed(a), embeddings.embed(b)
            sim = embeddings.cosine_similarity(va, vb)
            verdict = "✅ would be a semantic HIT" if sim >= threshold else "❌ would be a MISS"
            st.metric("Cosine similarity", f"{sim:.3f}")
            st.caption(f"{verdict} at threshold {threshold:.2f}. "
                       f"First 8 of {len(va)} numbers in A: {[round(float(x), 3) for x in va[:8]]}")

    with st.expander("Batch: send many questions at once"):
        st.caption("One question per line. Includes repeats and paraphrases on purpose; "
                   "they'll matter once the caches exist.")
        batch = st.text_area("Questions", SAMPLE_QUESTIONS, height=160)
        if st.button("Run batch"):
            lines = [q for q in batch.splitlines() if q.strip()]
            bar = st.progress(0.0)
            for i, q in enumerate(lines, 1):
                run(q)
                bar.progress(i / len(lines), text=f"{i}/{len(lines)}: {q}")
            st.success(f"Ran {len(lines)} questions. See the Dashboard tab.")

# ---------------- dashboard ----------------
with dash_tab:
    df = metrics.load()
    if df.empty:
        st.info("No requests yet. Ask something in the Ask tab.")
        st.stop()

    # Baseline ("none") runs can never hit, so mixing them in would drag the hit rate down.
    modes_seen = sorted(df["cache_mode"].unique())
    view = st.segmented_control("Show requests made with cache mode", ["all", *modes_seen],
                                default=df["cache_mode"].iloc[-1])  # the mode you used last
    if view and view != "all":
        df = df[df["cache_mode"] == view].reset_index(drop=True)
    df = df.copy()
    df["n"] = range(1, len(df) + 1)  # request # within this view
    s = metrics.summarize(df)

    r1 = st.columns(4)
    r1[0].metric("Requests", s["requests"])
    r1[1].metric("Cache hit rate", f"{s['hit_rate']:.0%}", help="hits / requests")
    r1[2].metric("Cache miss rate", f"{s['miss_rate']:.0%}", help="misses / requests")
    by_source = df["source"].value_counts()
    r1[3].metric("Exact / semantic hits",
                 f"{by_source.get('exact_cache', 0)} / {by_source.get('semantic_cache', 0)}",
                 help=f"{s['misses']} misses went to the LLM")

    r2 = st.columns(4)
    r2[0].metric("Avg latency, miss", f"{s['avg_miss_ms']:,.0f} ms")
    r2[1].metric("Avg latency, hit", f"{s['avg_hit_ms']:,.0f} ms" if s["hits"] else "–")
    r2[2].metric("Time saved", f"{s['time_saved_ms'] / 1000:,.1f} s",
                 help="hits × avg miss latency − actual hit latency")
    r2[3].metric("Tokens saved", f"{s['tokens_saved']:,}",
                 help=f"Tokens the LLM didn't have to generate. Generated so far: {s['tokens_generated']:,}")

    df["source_label"] = df["source"].map(SOURCE_LABELS)
    df["cum_hit_rate"] = df["cache_hit"].expanding().mean()

    left, right = st.columns(2)
    with left:
        st.subheader("Latency per request")
        latency = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("n:O", title="Request #"),
                y=alt.Y("latency_ms:Q", title="Latency (ms)"),
                color=source_color,
                tooltip=[
                    alt.Tooltip("n:O", title="Request #"),
                    alt.Tooltip("question:N", title="Question"),
                    alt.Tooltip("source_label:N", title="Source"),
                    alt.Tooltip("latency_ms:Q", title="Latency (ms)", format=",.0f"),
                    alt.Tooltip("load_ms:Q", title="Load (ms)", format=",.0f"),
                    alt.Tooltip("output_tokens:Q", title="Tokens"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(latency, width="stretch")

    with right:
        st.subheader("Cumulative cache hit rate")
        hit_rate = (
            alt.Chart(df)
            .mark_line(strokeWidth=2, color=colors[1], point=alt.OverlayMarkDef(size=64, filled=True))
            .encode(
                x=alt.X("n:O", title="Request #"),
                y=alt.Y("cum_hit_rate:Q", title="Hit rate so far",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                tooltip=[
                    alt.Tooltip("n:O", title="Request #"),
                    alt.Tooltip("cum_hit_rate:Q", title="Hit rate", format=".0%"),
                    alt.Tooltip("question:N", title="Question"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(hit_rate, width="stretch")

    sims = df.dropna(subset=["similarity"])
    if not sims.empty:
        st.subheader("Nearest-match similarity per request")
        st.caption("For every semantic lookup: how close the best stored question was. Above the dashed "
                   "threshold line it's a hit. Misses just under the line are the ones worth inspecting, "
                   "and so are hits that served the wrong answer.")
        points = (
            alt.Chart(sims)
            .mark_circle(size=110, opacity=1, stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("n:O", title="Request #"),
                y=alt.Y("similarity:Q", title="Cosine similarity", scale=alt.Scale(domain=[0, 1])),
                color=source_color,
                tooltip=[
                    alt.Tooltip("n:O", title="Request #"),
                    alt.Tooltip("question:N", title="Asked"),
                    alt.Tooltip("matched_question:N", title="Nearest stored"),
                    alt.Tooltip("similarity:Q", title="Similarity", format=".3f"),
                    alt.Tooltip("source_label:N", title="Result"),
                ],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"threshold": [threshold]}))
            .mark_rule(strokeDash=[6, 4], strokeWidth=2, color="gray")
            .encode(y="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="Current threshold")])
        )
        st.altair_chart((rule + points).properties(height=280), width="stretch")

    st.subheader("Request log")
    st.dataframe(
        df[["id", "ts", "question", "source", "cache_hit", "latency_ms", "load_ms",
            "output_tokens", "model", "similarity", "matched_question"]]
        .sort_values("id", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "ts": st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
            "load_ms": st.column_config.NumberColumn("Load (ms)", format="%.0f"),
            "cache_hit": st.column_config.CheckboxColumn("Hit"),
        },
    )

    st.subheader("Inside Redis")
    st.caption("Live from your Redis database, not the local log. "
               "Counters only move in cache modes other than `none`.")
    rs = redis_client.read_stats()
    reqs, hits = rs.get("requests", 0), rs.get("hits", 0)
    k = st.columns(4)
    k[0].metric("requests", reqs)
    k[1].metric("hits", hits)
    k[2].metric("misses", rs.get("misses", 0))
    k[3].metric("hit rate (hits / requests)", f"{hits / reqs:.0%}" if reqs else "–")
    st.code(f"HGETALL {redis_client.STATS_KEY}\n" + "\n".join(f"{f} → {v}" for f, v in rs.items()),
            language="text")

    cached = exact_cache.entries()
    st.markdown(f"**Exact-cache entries:** {len(cached)}  (`SCAN MATCH {exact_cache.PREFIX}:*`)")
    if cached:
        st.dataframe(
            pd.DataFrame(cached).sort_values("hits", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={"ttl_left_s": st.column_config.NumberColumn("TTL left (s)")},
        )

    sem_entries = semantic_cache.entries()
    try:
        info = redis_client.get_redis().execute_command("FT.INFO", semantic_cache.INDEX)
        info = info if isinstance(info, dict) else dict(zip(info[::2], info[1::2]))  # RESP3 vs RESP2
        num_docs = info.get("num_docs", "?")
    except Exception:
        num_docs = "index not created yet"
    st.markdown(f"**Semantic-cache entries:** {len(sem_entries)}  "
                f"(`FT.INFO {semantic_cache.INDEX}` → num_docs {num_docs})")
    if sem_entries:
        st.dataframe(
            pd.DataFrame(sem_entries).sort_values("hits", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={"ttl_left_s": st.column_config.NumberColumn("TTL left (s)")},
        )
