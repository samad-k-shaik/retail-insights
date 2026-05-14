"""
Multi-agent orchestration for retail sales analytics with LangGraph.

This module implements specialized agents:
- LanguageToQueryAgent: Converts natural language to DuckDB SQL
- DataExtractionAgent: Executes SQL and retrieves results
- ValidationAgent: Validates and formats agent outputs
- SummarizationAgent: Generates executive summaries
- RetailAgents: Orchestrates workflows using LangGraph

Key workflow patterns:
1. Summarization: Detect schema -> Compute KPIs -> Generate summary -> Validate
2. Q&A: Detect schema -> Convert to SQL -> Validate SQL -> Execute -> Format answer
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from agent_graph import build_qa_graph, build_summary_graph


def _strip_sql(sql: str) -> str:
    """Remove markdown code blocks from LLM-generated SQL.
    
    Handles both ```sql and ``` delimiters.
    """
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def _is_incomplete_text(text: str) -> bool:
    """Detect incomplete or truncated LLM responses.
    
    Returns True if text is too short, lacks sentence ending, or ends with
    common partial words (articles, conjunctions, helper verbs).
    """
    stripped = text.strip()
    if len(stripped.split()) < 25:
        return True
    if stripped[-1] not in ".!?":
        return True
    tail = stripped.lower().split()[-1].strip(".,;:")
    return tail in {"a", "an", "the", "and", "or", "of", "for", "to", "with", "by", "generated"}


def _fmt_number(value: float, label: str) -> str:
    """Format numbers with appropriate precision.
    
    Revenue fields use 2 decimal places; quantities use 0 decimals.
    """
    if label == "revenue":
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def _series_leader_text(grouped: pd.Series, role: str, metric_label: str, strongest: bool = True) -> str | None:
    """Generate natural language text for top/bottom performing segments.
    
    Args:
        grouped: Series with aggregated metrics
        role: Schema role (e.g., "category", "region")
        metric_label: Metric name (e.g., "revenue")
        strongest: If True, get leader; if False, get lowest performer
        
    Returns:
        Human-readable comparison sentence or None if series is empty
    """
    if grouped.empty:
        return None
    item = grouped.index[0] if strongest else grouped.index[-1]
    value = float(grouped.iloc[0] if strongest else grouped.iloc[-1])
    direction = "leads" if strongest else "is lowest"
    return f"{role} '{item}' {direction} by {metric_label} with {_fmt_number(value, metric_label)}"


def _top_bottom_lines(grouped: pd.Series, role: str, metric_label: str, limit: int = 3) -> tuple[list[str], list[str]]:
    """Extract top and bottom performing items from a grouped series.
    
    Args:
        grouped: Series with aggregated metrics
        role: Schema role (e.g., "category", "region")
        metric_label: Metric name (e.g., "revenue")
        limit: Number of top/bottom items to return
        
    Returns:
        Tuple of (top_items_list, bottom_items_list)
    """
    if grouped.empty:
        return [], []
    top = [
        f"{role} '{idx}' = {_fmt_number(float(value), metric_label)}"
        for idx, value in grouped.head(limit).items()
    ]
    bottom = [
        f"{role} '{idx}' = {_fmt_number(float(value), metric_label)}"
        for idx, value in grouped.tail(limit).items()
    ]
    return top, bottom


@dataclass
class AgentResult:
    """Container for agent output with metadata.
    
    Attributes:
        name: Agent identifier
        output: Generated output (SQL, summary, answer, etc.)
    """
    name: str
    output: str


class LanguageToQueryAgent:
    name = "language_to_query_resolution_agent"

    def __init__(self, llm: Any):
        self.llm = llm

    def run(self, question: str, profile: dict, conversation_context: str = "") -> AgentResult:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You translate retail analytics questions into DuckDB SQL for a table named sales. "
                    "Return only one read-only SELECT statement. Use double quotes around columns only if needed. "
                    "Prefer aggregates over SELECT * and include a LIMIT for detail queries. "
                    "Use detected_schema_roles to choose the right original columns for the uploaded dataset. "
                    "Use helper columns when available: __ri_revenue for sales/amount, __ri_quantity for units/qty, "
                    "and __ri_date for dates. "
                    "For YoY growth, compare matching periods across years with (current - previous) / previous."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dataset profile:\n{profile}\n\n"
                    f"Conversation context:\n{conversation_context or 'None'}\n\n"
                    f"Question: {question}\nSQL:"
                ),
            },
        ]
        sql = _strip_sql(self.llm.chat(prompt, temperature=0.0, max_tokens=512))
        return AgentResult(self.name, sql)


class DataExtractionAgent:
    name = "data_extraction_agent"

    def __init__(self, data_layer: Any):
        self.data_layer = data_layer

    def run(self, sql: str) -> tuple[AgentResult, pd.DataFrame]:
        result_df = self.data_layer.run_sql(sql)
        preview = result_df.head(30).to_csv(index=False)
        return AgentResult(self.name, preview), result_df


class ValidationAgent:
    name = "validation_agent"

    def __init__(self, llm: Any):
        self.llm = llm

    def run(self, question: str, sql: str, result_df: pd.DataFrame) -> AgentResult:
        if result_df.empty:
            return AgentResult(
                self.name,
                "No matching rows were found for the generated query. Try broadening the filters or date range.",
            )

        result_text = result_df.head(30).to_csv(index=False)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a retail business analyst and validation agent. Answer only from the SQL result. "
                    "If the result does not support the question, say what is missing. Keep the response concise, "
                    "include the main number when available, and add one practical recommendation."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nSQL used: {sql}\nResult CSV:\n{result_text}\nAnswer:",
            },
        ]
        answer = self.llm.chat(prompt, temperature=0.1, max_tokens=512)
        return AgentResult(self.name, answer)


class SummarizationAgent:
    name = "summarization_agent"

    def __init__(self, llm: Any, data_layer: Any):
        self.llm = llm
        self.data_layer = data_layer

    def run(self, df: pd.DataFrame) -> AgentResult:
        prepared = self.data_layer.prepare_dataframe(df)
        profile = self.data_layer.profile_dataframe(df)
        roles = profile["detected_schema_roles"]
        summary_parts = []
        findings = []
        highlight_lines = []
        top_segment_lines = []
        weak_segment_lines = []
        geography_lines = []
        operations_lines = []
        date_range_text = None

        metric_col = "__ri_revenue" if "__ri_revenue" in prepared.columns else None
        metric_label = "revenue"
        if metric_col is None and "__ri_quantity" in prepared.columns:
            metric_col = "__ri_quantity"
            metric_label = "quantity"

        summary_parts.append(f"Detected schema roles: {roles}")
        summary_parts.append(f"Rows analyzed: {len(prepared):,}")

        if metric_col:
            total_metric = prepared[metric_col].sum()
            summary_parts.append(f"Total {metric_label}: {total_metric:,.2f}.")
            findings.append(f"Total {metric_label} is {_fmt_number(total_metric, metric_label)} across {len(prepared):,} records.")
            highlight_lines.append(f"Total {metric_label}: {_fmt_number(float(total_metric), metric_label)}")
        else:
            findings.append(f"The dataset contains {len(prepared):,} records; no clear revenue or quantity metric was detected.")
            highlight_lines.append(f"Total records: {len(prepared):,}")

        if "__ri_quantity" in prepared.columns:
            total_quantity = float(prepared["__ri_quantity"].fillna(0).sum())
            highlight_lines.append(f"Total quantity: {_fmt_number(total_quantity, 'quantity')}")

        order_col = roles.get("order_id")
        if order_col and order_col in prepared.columns:
            highlight_lines.append(f"Unique orders: {prepared[order_col].nunique(dropna=True):,}")

        top_findings = []
        weak_findings = []
        operational_findings = []
        summary_roles = [
            "region",
            "city",
            "state",
            "country",
            "category",
            "product",
            "sku",
            "status",
            "channel",
            "fulfillment",
            "ship_service_level",
            "courier_status",
            "size",
            "b2b",
            "fulfilled_by",
        ]
        for role in summary_roles:
            column = roles.get(role)
            if column and column in prepared.columns:
                if metric_col:
                    grouped = prepared.groupby(column)[metric_col].sum().sort_values(ascending=False).head(10)
                    summary_parts.append(f"Top {role} by {metric_label}:\n{grouped.to_csv(header=True)}")
                    strongest = _series_leader_text(grouped, role, metric_label, strongest=True)
                    weakest = _series_leader_text(grouped, role, metric_label, strongest=False)
                    top_lines, bottom_lines = _top_bottom_lines(grouped, role, metric_label)
                    if strongest:
                        top_findings.append(strongest)
                    if role in {"category", "product", "sku", "channel", "fulfillment", "size"}:
                        top_segment_lines.extend(top_lines)
                    if role in {"region", "state", "city", "country"}:
                        geography_lines.extend(top_lines[:2])
                    if weakest and len(grouped) > 1 and role in {"category", "product", "sku", "region", "state", "city", "fulfillment"}:
                        weak_findings.append(weakest)
                        weak_segment_lines.extend(bottom_lines)
                else:
                    grouped = prepared[column].value_counts().head(10)
                    summary_parts.append(f"Top {role} by order count:\n{grouped.to_csv(header=True)}")
                    if not grouped.empty:
                        top_findings.append(f"{role} '{grouped.index[0]}' leads by record count with {int(grouped.iloc[0]):,}")
                        count_lines = [f"{role} '{idx}' = {int(value):,} records" for idx, value in grouped.head(3).items()]
                        if role in {"category", "product", "sku", "channel", "fulfillment", "size"}:
                            top_segment_lines.extend(count_lines)
                        if role in {"region", "state", "city", "country"}:
                            geography_lines.extend(count_lines[:2])
                        if len(grouped) > 1 and role in {"category", "product", "sku", "region", "state", "city", "fulfillment"}:
                            weak_findings.append(f"{role} '{grouped.index[-1]}' is lowest by record count with {int(grouped.iloc[-1]):,}")
                            weak_segment_lines.extend(
                                [f"{role} '{idx}' = {int(value):,} records" for idx, value in grouped.tail(3).items()]
                            )

        if "__ri_date" in prepared.columns:
            dated = prepared.dropna(subset=["__ri_date"])
            if not dated.empty:
                date_range_text = f"{dated['__ri_date'].min().date()} to {dated['__ri_date'].max().date()}"
                if metric_col:
                    monthly = dated.groupby(dated["__ri_date"].dt.to_period("M"))[metric_col].sum().tail(6)
                    summary_parts.append(f"Recent monthly {metric_label}:\n{monthly.to_csv(header=True)}")
                    if len(monthly) >= 2:
                        latest = float(monthly.iloc[-1])
                        previous = float(monthly.iloc[-2])
                        direction = "up" if latest >= previous else "down"
                        findings.append(
                            f"Latest monthly {metric_label} is {direction} versus the prior month "
                            f"({_fmt_number(latest, metric_label)} vs {_fmt_number(previous, metric_label)})."
                        )
                else:
                    monthly = dated.groupby(dated["__ri_date"].dt.to_period("M")).size().tail(6)
                    summary_parts.append(f"Recent monthly order count:\n{monthly.to_csv(header=True)}")
                    if len(monthly) >= 2:
                        direction = "up" if monthly.iloc[-1] >= monthly.iloc[-2] else "down"
                        findings.append(
                            f"Latest monthly order count is {direction} versus the prior month "
                            f"({int(monthly.iloc[-1]):,} vs {int(monthly.iloc[-2]):,})."
                        )

        status_col = roles.get("status")
        risk_text = "Review the lowest-performing categories, products, or fulfillment paths for pricing, availability, and operations issues."
        if status_col and status_col in prepared.columns:
            status_counts = prepared[status_col].astype(str).str.lower().value_counts()
            cancelled = sum(count for status, count in status_counts.items() if "cancel" in status)
            if cancelled:
                rate = cancelled / max(len(prepared), 1) * 100
                risk_text = f"Cancellation-related records are {cancelled:,} ({rate:.1f}%); investigate stock, delivery, or listing quality causes."
                operational_findings.append(risk_text)
                operations_lines.append(risk_text)
            top_status = prepared[status_col].astype(str).value_counts().head(5)
            if not top_status.empty:
                operations_lines.extend([f"status '{idx}' = {int(value):,} records" for idx, value in top_status.items()])

        courier_col = roles.get("courier_status")
        if courier_col and courier_col in prepared.columns:
            courier_counts = prepared[courier_col].astype(str).value_counts().head(3)
            if not courier_counts.empty:
                courier_text = f"Courier status is led by '{courier_counts.index[0]}' with {int(courier_counts.iloc[0]):,} records."
                operational_findings.append(courier_text)
                operations_lines.append(courier_text)

        fulfillment_col = roles.get("fulfillment")
        if fulfillment_col and fulfillment_col in prepared.columns:
            fulfillment_counts = prepared[fulfillment_col].astype(str).value_counts().head(3)
            if not fulfillment_counts.empty:
                fulfillment_text = f"Fulfillment is concentrated in '{fulfillment_counts.index[0]}' with {int(fulfillment_counts.iloc[0]):,} records."
                operational_findings.append(fulfillment_text)
                operations_lines.append(fulfillment_text)

        driver_text = "; ".join(top_findings[:3]) if top_findings else "The available categorical fields do not show a clear driver."
        weak_text = "; ".join(weak_findings[:2]) if weak_findings else "No clear weak segment was detected from the available fields."
        operations_text = "; ".join(operational_findings[:2]) if operational_findings else "No major operational exception was detected from status, courier, or fulfillment fields."
        trend_text = findings[1] if len(findings) > 1 else "Trend analysis needs more than one detected time period."
        date_text = f" Date coverage is {date_range_text}." if date_range_text else ""
        fallback_summary = "\n".join(
            [
                "## Detailed File Summary",
                f"- File overview: Analyzed {len(prepared):,} rows and {len(prepared.columns):,} usable columns.{date_text}",
                f"- KPI highlights: {'; '.join(highlight_lines[:5])}.",
                f"- Revenue or volume trend: {trend_text}",
                f"- Primary business drivers: {driver_text}.",
                f"- Top segment details: {'; '.join(top_segment_lines[:8]) or 'No segment detail available.'}.",
                f"- Underperformance signals: {weak_text}. Additional low segments: {'; '.join(weak_segment_lines[:5]) or 'None detected'}.",
                f"- Geography highlights: {'; '.join(geography_lines[:6]) or 'No geography fields were detected.'}.",
                f"- Operations highlights: {operations_text}. Details: {'; '.join(operations_lines[:6]) or 'No operational status fields were detected.'}.",
                f"- Recommended actions: {risk_text} Prioritize the leading segments for stock and campaign planning, and review weak segments for pricing, availability, or delivery issues.",
                "",
                "## Highlight Points",
                *[f"- {line}" for line in (highlight_lines[:4] + top_segment_lines[:4] + weak_segment_lines[:2] + operations_lines[:2])],
            ]
        )

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an executive retail insights assistant. The uploaded dataset schema may vary. "
                    "Use the detected schema and computed aggregates below to summarize the whole uploaded file. "
                    "Return a detailed business summary with sections named Detailed File Summary and Highlight Points. "
                    "Cover KPI totals, date coverage, top segments, weak segments, geography, operations, and recommended actions. "
                    "If revenue is unavailable, summarize order volume/counts instead. Do not invent unsupported facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Baseline summary to preserve:\n{fallback_summary}\n\n"
                    f"Computed aggregates:\n{chr(10).join(summary_parts) or 'No usable metrics were available.'}"
                ),
            },
        ]
        try:
            answer = self.llm.chat(prompt, temperature=0.1, max_tokens=1600)
        except Exception:
            answer = fallback_summary
        if _is_incomplete_text(answer):
            answer = fallback_summary
        return AgentResult(self.name, answer)


class RetailAgents:
    def __init__(self, llm: Any, data_layer: Any):
        self.llm = llm
        self.data_layer = data_layer
        self.language_agent = LanguageToQueryAgent(llm)
        self.extraction_agent = DataExtractionAgent(data_layer)
        self.validation_agent = ValidationAgent(llm)
        self.summarization_agent = SummarizationAgent(llm, data_layer)
        self.last_trace: list[AgentResult] = []
        self.summary_graph = build_summary_graph(
            {
                "schema_detection_node": self._summary_schema_node,
                "kpi_insight_node": self._summary_kpi_node,
                "summary_generation_node": self._summary_generation_node,
                "summary_validation_node": self._summary_validation_node,
            }
        )
        self.qa_graph = build_qa_graph(
            {
                "schema_detection_node": self._qa_schema_node,
                "language_to_query_node": self._qa_language_node,
                "sql_validation_node": self._qa_sql_validation_node,
                "data_extraction_node": self._qa_extraction_node,
                "validation_answer_node": self._qa_validation_node,
            }
        )

    def summarize_performance(self, df: pd.DataFrame) -> str:
        state = self.summary_graph.invoke({"df": df, "trace": []})
        self.last_trace = state.get("trace", [])
        return state.get("answer", "")

    def handle_question(self, df: pd.DataFrame, question: str, conversation_context: str = "") -> str:
        state = self.qa_graph.invoke(
            {
                "df": df,
                "question": question,
                "conversation_context": conversation_context,
                "trace": [],
            }
        )
        self.last_trace = state.get("trace", [])
        return state.get("answer", "")

    def _summary_schema_node(self, state: dict) -> dict:
        profile = self.data_layer.profile_dataframe(state["df"])
        trace = state.get("trace", []) + [
            AgentResult("schema_detection_node", str(profile.get("detected_schema_roles", {})))
        ]
        return {**state, "profile": profile, "trace": trace}

    def _summary_kpi_node(self, state: dict) -> dict:
        kpis = self.data_layer.compute_kpis(state["df"])
        trace = state.get("trace", []) + [AgentResult("kpi_insight_node", str(kpis))]
        return {**state, "kpis": kpis, "trace": trace}

    def _summary_generation_node(self, state: dict) -> dict:
        result = self.summarization_agent.run(state["df"])
        trace = state.get("trace", []) + [AgentResult("summary_generation_node", result.output)]
        return {**state, "answer": result.output, "trace": trace}

    def _summary_validation_node(self, state: dict) -> dict:
        answer = state.get("answer", "")
        validation = "Summary accepted."
        if _is_incomplete_text(answer):
            result = self.summarization_agent.run(state["df"])
            answer = result.output
            validation = "Summary regenerated because previous response was incomplete."
        trace = state.get("trace", []) + [AgentResult("summary_validation_node", validation)]
        return {**state, "answer": answer, "trace": trace}

    def _qa_schema_node(self, state: dict) -> dict:
        self.data_layer.register_df(state["df"], table_name="sales")
        profile = self.data_layer.profile_dataframe(state["df"])
        trace = state.get("trace", []) + [
            AgentResult("schema_detection_node", str(profile.get("detected_schema_roles", {})))
        ]
        return {**state, "profile": profile, "trace": trace}

    def _qa_language_node(self, state: dict) -> dict:
        query_result = self.language_agent.run(
            state["question"],
            state["profile"],
            state.get("conversation_context", ""),
        )
        trace = state.get("trace", []) + [AgentResult("language_to_query_node", query_result.output)]
        return {**state, "sql": query_result.output, "trace": trace}

    def _qa_sql_validation_node(self, state: dict) -> dict:
        sql = state["sql"]
        validation = "SQL validation passed."
        try:
            self.data_layer.validate_read_only_sql(sql)
        except Exception as exc:
            sql = "SELECT * FROM sales LIMIT 20"
            validation = f"Generated SQL failed validation: {exc}. Fallback SQL selected."
        trace = state.get("trace", []) + [AgentResult("sql_validation_node", validation)]
        return {**state, "sql": sql, "trace": trace}

    def _qa_extraction_node(self, state: dict) -> dict:
        sql = state["sql"]
        try:
            extraction_result, result_df = self.extraction_agent.run(sql)
        except Exception as exc:
            sql = "SELECT * FROM sales LIMIT 20"
            extraction_result, result_df = self.extraction_agent.run(sql)
            extraction_result = AgentResult(
                "data_extraction_node",
                f"Fallback used because extraction failed: {exc}\n\n{extraction_result.output}",
            )
        trace = state.get("trace", []) + [AgentResult("data_extraction_node", extraction_result.output)]
        return {**state, "sql": sql, "result_df": result_df, "trace": trace}

    def _qa_validation_node(self, state: dict) -> dict:
        validation_result = self.validation_agent.run(state["question"], state["sql"], state["result_df"])
        trace = state.get("trace", []) + [AgentResult("validation_answer_node", validation_result.output)]
        return {**state, "answer": validation_result.output, "trace": trace}

    def handle_question_legacy(self, df: pd.DataFrame, question: str, conversation_context: str = "") -> str:
        self.data_layer.register_df(df, table_name="sales")
        profile = self.data_layer.profile_dataframe(df)
        query_result = self.language_agent.run(question, profile, conversation_context)
        sql = query_result.output

        try:
            extraction_result, result_df = self.extraction_agent.run(sql)
        except Exception as exc:
            fallback = "SELECT * FROM sales LIMIT 20"
            extraction_result, result_df = self.extraction_agent.run(fallback)
            sql = fallback
            query_result = AgentResult(query_result.name, f"{query_result.output}\nFallback used because: {exc}")

        validation_result = self.validation_agent.run(question, sql, result_df)
        self.last_trace = [query_result, extraction_result, validation_result]
        return validation_result.output
