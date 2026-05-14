import os

import pandas as pd
import streamlit as st

from agents import RetailAgents
from data_layer import DataLayer
from llm_client import LLMClient


st.set_page_config(page_title="Retail Insights Assistant", layout="wide", page_icon="RI")


@st.cache_data(ttl=300, show_spinner=False)
def cached_list_gcs_files(bucket_name: str) -> list[dict]:
    """Cache GCS bucket listings for a short time."""
    return data_layer.list_gcs_files(bucket_name)


@st.cache_data(ttl=300, show_spinner=False)
def cached_load_gcs_csv(gcs_uri: str) -> pd.DataFrame:
    """Cache loaded GCS CSV files for repeated access during a session."""
    return data_layer.load_csv_from_gcs(gcs_uri)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1440px;}
    .app-title {font-size: 2.1rem; font-weight: 750; margin-bottom: .15rem;}
    .app-subtitle {color: #64748b; font-size: 1rem; margin-bottom: 1.25rem;}
    .kpi-card {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 18px;
        min-height: 112px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.16);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: .82rem;
        font-weight: 650;
        margin-bottom: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.35rem;
        line-height: 1.22;
        font-weight: 760;
        overflow-wrap: anywhere;
    }
    .kpi-detail {
        color: #cbd5e1;
        font-size: .78rem;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }
    .section-note {color: #64748b; font-size: .92rem;}
    .status-pill {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 3px 10px;
        margin-right: 6px;
        color: #334155;
        font-size: .82rem;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_input(data_layer: DataLayer) -> tuple[pd.DataFrame | None, str]:
    """Load input dataset from upload or GCS with dropdown for GCS files."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
    if "gcs_bucket" not in st.session_state:
        st.session_state.gcs_bucket = os.getenv("DEFAULT_GCS_BUCKET", "datsets_blend")
    if "gcs_uri" not in st.session_state:
        st.session_state.gcs_uri = ""
    if "gcs_loaded" not in st.session_state:
        st.session_state.gcs_loaded = False

    input_mode = st.sidebar.selectbox(
        "Input type",
        ["Upload file", "GCS URI"],
        index=1,
        key="input_type",
        help="Select GCS URI to load shared datasets from cloud storage.",
    )
    st.sidebar.caption(
        "For shared use, choose GCS URI so your manager can load files from a shared bucket instead of uploading locally."
    )

    if input_mode == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV, Excel, JSON, or text report",
            type=["csv", "xlsx", "json", "txt"],
        )
        if uploaded is not None:
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded)
            elif name.endswith(".xlsx"):
                st.session_state.df = pd.read_excel(uploaded)
            elif name.endswith(".json"):
                st.session_state.df = pd.read_json(uploaded)
            else:
                st.session_state.report_text = uploaded.getvalue().decode("utf-8", errors="replace")
                st.session_state.df = pd.DataFrame({"report_text": [st.session_state.report_text]})
            st.session_state.gcs_loaded = False
    else:
        # GCS mode with dropdown
        st.sidebar.write("**GCS Bucket Files**")
        bucket_name = st.sidebar.text_input(
            "GCS Bucket",
            value=st.session_state.gcs_bucket,
            key="gcs_bucket",
            help="Shared GCS bucket name (no gs:// prefix). If the bucket exists, the app will list files automatically.",
        )
        st.sidebar.caption(
            "If the app is deployed to Cloud Run with the correct service account, shared files can be loaded directly from GCS."
        )

        # List files from bucket
        try:
            with st.spinner("Loading files from GCS..."):
                gcs_files = cached_list_gcs_files(bucket_name)

            if not gcs_files:
                st.sidebar.warning(f"No files found in gs://{bucket_name}")
            else:
                # Create dropdown with file names and sizes
                file_options = [f"{f['name']} ({f['size_mb']} MB)" for f in gcs_files]
                selected_file = st.sidebar.selectbox("Select file", file_options, key="gcs_file_select")

                if selected_file:
                    selected_idx = file_options.index(selected_file)
                    gcs_uri = gcs_files[selected_idx]["uri"]

                    # Show the URI for reference
                    st.sidebar.text(f"URI: {gcs_uri}")

                    if st.sidebar.button("Load from GCS", type="primary", key="load_gcs_button"):
                        with st.spinner("Loading CSV from GCS..."):
                            st.session_state.df = cached_load_gcs_csv(gcs_uri)
                            st.session_state.report_text = ""
                            st.session_state.gcs_uri = gcs_uri
                            st.session_state.gcs_loaded = True
                            st.sidebar.success(f"✓ Loaded {len(st.session_state.df):,} rows")
                    elif st.session_state.gcs_uri == gcs_uri and st.session_state.gcs_loaded:
                        st.sidebar.success(f"✓ Loaded {len(st.session_state.df):,} rows from cached GCS file")
        except Exception as e:
            st.sidebar.error(
                "Error accessing GCS: {}. "
                "Check that the bucket exists, the Cloud Run service account has storage permission, "
                "and the bucket name is correct."
                .format(str(e))
            )

            # Fallback to manual URI entry
            st.sidebar.write("**Or enter URI manually:**")
            manual_gcs_uri = st.sidebar.text_input(
                "GCS URI",
                placeholder="gs://bucket/path/file.csv",
                key="manual_gcs_uri",
            )
            if manual_gcs_uri and st.sidebar.button("Load URI", key="load_manual_gcs_button"):
                with st.spinner("Loading CSV from GCS..."):
                    st.session_state.df = cached_load_gcs_csv(manual_gcs_uri)
                    st.session_state.report_text = ""
                    st.session_state.gcs_uri = manual_gcs_uri
                    st.session_state.gcs_loaded = True

    return st.session_state.df, st.session_state.report_text



def build_llm() -> LLMClient:
    st.sidebar.header("LLM")
    llm_provider = st.sidebar.selectbox("Provider", ["vertex_ai", "openai"], index=0)
    vertex_project = st.sidebar.text_input(
        "Vertex project",
        value=os.getenv("GOOGLE_CLOUD_PROJECT", "blend360-496117"),
    )
    vertex_location = st.sidebar.text_input(
        "Vertex location",
        value=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    vertex_model = st.sidebar.text_input(
        "Vertex model",
        value=os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
    )
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    return LLMClient(
        provider=llm_provider,
        project=vertex_project or None,
        location=vertex_location or None,
        model=vertex_model or None,
    )


def render_kpi_cards(cards: list[dict], max_cards: int = 8) -> None:
    cards = cards[:max_cards]
    if not cards:
        st.info("No KPI cards could be generated from the detected schema.")
        return
    for start in range(0, len(cards), 4):
        columns = st.columns(min(4, len(cards) - start))
        for column, card in zip(columns, cards[start : start + 4]):
            value = str(card["value"])
            display_value = value if len(value) <= 26 else value[:23] + "..."
            detail = value if display_value != value else card.get("detail", "")
            column.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{card["label"]}</div>
                    <div class="kpi-value">{display_value}</div>
                    <div class="kpi-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_charts(data_layer: DataLayer, df: pd.DataFrame) -> None:
    prepared = data_layer.prepare_dataframe(df)
    schema = data_layer.infer_schema(df)
    metric = "__ri_revenue" if "__ri_revenue" in prepared.columns else "__ri_quantity" if "__ri_quantity" in prepared.columns else None

    chart_cols = st.columns(2)
    with chart_cols[0]:
        category_col = schema.get("category")
        if category_col and category_col in prepared.columns:
            if metric:
                category = prepared.groupby(category_col)[metric].sum().sort_values(ascending=False).head(10)
            else:
                category = prepared[category_col].value_counts().head(10)
            st.caption("Top categories")
            st.bar_chart(category)
        else:
            st.caption("Top categories")
            st.info("Category column not detected.")

    with chart_cols[1]:
        state_col = schema.get("state") or schema.get("region") or schema.get("city")
        if state_col and state_col in prepared.columns:
            if metric:
                geography = prepared.groupby(state_col)[metric].sum().sort_values(ascending=False).head(10)
            else:
                geography = prepared[state_col].value_counts().head(10)
            st.caption(f"Top geography: {state_col}")
            st.bar_chart(geography)
        else:
            st.caption("Top geography")
            st.info("Location column not detected.")

    if "__ri_date" in prepared.columns:
        dated = prepared.dropna(subset=["__ri_date"])
        if not dated.empty:
            st.caption("Monthly trend")
            if metric:
                trend = dated.groupby(dated["__ri_date"].dt.to_period("M").astype(str))[metric].sum()
            else:
                trend = dated.groupby(dated["__ri_date"].dt.to_period("M").astype(str)).size()
            st.line_chart(trend)


def render_trace(trace) -> None:
    for step in trace:
        st.markdown(f"**{step.name}**")
        st.code(step.output)


data_layer = DataLayer()

st.sidebar.header("Data")
df, report_text = load_input(data_layer)

st.markdown('<div class="app-title">Retail Insights Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">GenAI retail analytics with LangGraph agents, schema detection, KPI cards, and natural-language Q&A.</div>',
    unsafe_allow_html=True,
)

if df is None:
    st.info("Upload a dataset or provide a GCS CSV path to begin.")
    st.stop()

try:
    llm = build_llm()
except Exception as exc:
    st.error(f"LLM setup failed: {exc}")
    st.stop()

agents = RetailAgents(llm=llm, data_layer=data_layer)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

profile = data_layer.profile_dataframe(df)
kpis = {} if report_text else data_layer.compute_kpis(df)

st.markdown(
    " ".join(
        [
            f'<span class="status-pill">{len(df):,} rows</span>',
            f'<span class="status-pill">{len(df.columns):,} columns</span>',
            '<span class="status-pill">LangGraph orchestration</span>',
            '<span class="status-pill">Vertex AI Gemini</span>',
        ]
    ),
    unsafe_allow_html=True,
)

dashboard_tab, summary_tab, chat_tab, data_tab = st.tabs(["Dashboard", "Summary", "Chat", "Data & Schema"])

with dashboard_tab:
    st.subheader("KPI Snapshot")
    if report_text:
        st.text_area("Report preview", report_text[:5000], height=260)
    else:
        render_kpi_cards(kpis.get("cards", []))
        st.subheader("Auto Insights")
        insights = kpis.get("insights", [])
        if insights:
            for insight in insights[:8]:
                st.markdown(f"- {insight}")
        else:
            st.info("No automatic insights were generated from the detected schema.")
        st.subheader("Performance Views")
        render_charts(data_layer, df)

with summary_tab:
    st.subheader("Detailed File Summary")
    st.markdown('<p class="section-note">Summarizes the whole uploaded file using detected schema, KPI totals, top and weak segments, geography, operations, and highlight points.</p>', unsafe_allow_html=True)
    if st.button("Generate Detailed Summary", type="primary"):
        with st.spinner("Running LangGraph summary workflow..."):
            summary = agents.summarize_performance(df)
        st.markdown(summary)
        with st.expander("Agent trace"):
            render_trace(agents.last_trace)

with chat_tab:
    st.subheader("Conversational Q&A")
    st.markdown('<p class="section-note">Questions are converted into validated DuckDB SQL, executed, then checked by the validation agent.</p>', unsafe_allow_html=True)
    suggested = [
        "Which category generated the highest revenue?",
        "How many orders were cancelled?",
        "Which state generated the highest revenue?",
        "Which fulfillment method has the most shipped orders?",
    ]
    selected = st.selectbox("Try a sample question", [""] + suggested)
    question = st.text_input("Ask a business question", value=selected)
    if st.button("Ask", type="primary") and question:
        with st.spinner("Running LangGraph Q&A workflow..."):
            context = "\n".join(
                f"Q: {item['question']}\nA: {item['answer']}" for item in st.session_state.chat_history[-3:]
            )
            answer = agents.handle_question(df, question, conversation_context=context)
            st.session_state.chat_history.append({"question": question, "answer": answer})
        st.markdown("#### Answer")
        st.write(answer)
        with st.expander("Agent trace"):
            render_trace(agents.last_trace)

    if st.session_state.chat_history:
        st.markdown("#### Recent Conversation")
        for item in reversed(st.session_state.chat_history[-5:]):
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.write(item["answer"])

with data_tab:
    st.subheader("Data Preview")
    if report_text:
        st.text_area("Report preview", report_text[:5000], height=260)
    else:
        st.dataframe(df.head(200), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Detected Schema Roles")
        st.json(profile["detected_schema_roles"])
    with col_b:
        st.subheader("Profile")
        st.json(
            {
                "rows": profile["rows"],
                "columns": profile["original_columns"],
                "date_min": profile.get("date_min"),
                "date_max": profile.get("date_max"),
                "helper_columns": profile["helper_columns"],
            }
        )
