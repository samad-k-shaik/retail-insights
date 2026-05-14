from pptx import Presentation
from typing import Any


def set_title(slide: Any, title: str) -> None:
    title_shape = slide.shapes.title
    if title_shape is not None:
        title_shape.text = title


def get_body_placeholder(slide: Any) -> Any:
    return slide.shapes.placeholders[1]


def add_bullet_slide(prs: Presentation, title: str, intro: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(slide, title)
    body = get_body_placeholder(slide)
    text_frame = body.text_frame
    text_frame.text = intro
    for bullet in bullets:
        text_frame.add_paragraph().text = bullet


def make_architecture_slides(output_path='architecture.pptx'):
    prs = Presentation()
    add_bullet_slide(
        prs,
        "Retail Insights Assistant - Architecture",
        "System components:",
        [
            "- Inputs: CSV, Excel, JSON, text report, or GCS CSV",
            "- MVP analytics: Pandas + DuckDB",
            "- Orchestration: LangGraph stateful agent workflow",
            "- LLM: Vertex AI Gemini via gcloud ADC",
            "- UI: Streamlit dashboard with KPI, summary, chat, and schema tabs",
        ],
    )
    add_bullet_slide(
        prs,
        "Agentic Query Pipeline",
        "Three required agents:",
        [
            "- LangGraph schema detection node profiles each uploaded dataset",
            "- Language-to-query node converts natural language to read-only SQL",
            "- Data extraction node validates and executes SQL",
            "- Validation node answers only from returned data",
            "- Conversation context is passed into the SQL planning prompt",
        ],
    )
    add_bullet_slide(
        prs,
        "Scaling Notes",
        "Key scaling decisions:",
        [
            "- Land raw files in GCS and convert to partitioned Parquet/Delta",
            "- Use Dataflow, Dataproc, or BigQuery for 100GB+ preprocessing",
            "- Query partitioned BigQuery tables and materialized aggregates",
            "- Use vector DB for report retrieval and cache frequent LLM/query results",
        ],
    )
    add_bullet_slide(
        prs,
        "Monitoring and Evaluation",
        "Operational controls:",
        [
            "- Track SQL validity, row counts, answer accuracy, p95 latency, and cost/query",
            "- Fall back to raw aggregates when validation confidence is low",
            "- Log prompts, generated SQL, and outcomes for test-set evaluation",
        ],
    )

    prs.save(output_path)
    print('Saved slides to', output_path)

if __name__ == '__main__':
    make_architecture_slides()
