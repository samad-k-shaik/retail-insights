"""
LangGraph orchestration for multi-agent retail analytics workflows.

This module defines two agent graphs:
1. SummaryGraph: Generates executive summaries from uploaded datasets
2. QAGraph: Converts natural language questions to SQL and answers them
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph


class SummaryGraphState(TypedDict, total=False):
    """State schema for summarization workflow.
    
    Attributes:
        df: Input dataset as pandas DataFrame
        profile: Detected schema roles and dataset metadata
        kpis: Computed key performance indicators
        answer: Generated executive summary
        trace: Audit trail of agent execution steps
    """
    df: pd.DataFrame
    profile: dict
    kpis: dict
    answer: str
    trace: list[Any]


class QAGraphState(TypedDict, total=False):
    """State schema for question-answering workflow.
    
    Attributes:
        df: Input dataset as pandas DataFrame
        question: User's natural language question
        conversation_context: Previous Q&A for context
        profile: Detected schema roles and dataset metadata
        sql: Generated DuckDB SQL query
        result_df: Query result as DataFrame
        answer: Validation agent's response
        trace: Audit trail of agent execution steps
    """
    df: pd.DataFrame
    question: str
    conversation_context: str
    profile: dict
    sql: str
    result_df: pd.DataFrame
    answer: str
    trace: list[Any]


def build_summary_graph(nodes: dict[str, Callable[[dict], dict]]):
    """Construct the summary generation workflow graph.
    
    Flow:
        START -> schema_detection -> kpi_insight -> summary_generation -> 
        summary_validation -> END
    
    Args:
        nodes: Dictionary mapping node names to callable handlers
        
    Returns:
        Compiled LangGraph graph ready for execution
    """
    graph = StateGraph(SummaryGraphState)
    graph.add_node("schema_detection_node", nodes["schema_detection_node"])
    graph.add_node("kpi_insight_node", nodes["kpi_insight_node"])
    graph.add_node("summary_generation_node", nodes["summary_generation_node"])
    graph.add_node("summary_validation_node", nodes["summary_validation_node"])

    graph.add_edge(START, "schema_detection_node")
    graph.add_edge("schema_detection_node", "kpi_insight_node")
    graph.add_edge("kpi_insight_node", "summary_generation_node")
    graph.add_edge("summary_generation_node", "summary_validation_node")
    graph.add_edge("summary_validation_node", END)
    return graph.compile()


def build_qa_graph(nodes: dict[str, Callable[[dict], dict]]):
    """Construct the question-answering workflow graph.
    
    Flow:
        START -> schema_detection -> language_to_query -> sql_validation -> 
        data_extraction -> validation_answer -> END
    
    Args:
        nodes: Dictionary mapping node names to callable handlers
        
    Returns:
        Compiled LangGraph graph ready for execution
    """
    graph = StateGraph(QAGraphState)
    graph.add_node("schema_detection_node", nodes["schema_detection_node"])
    graph.add_node("language_to_query_node", nodes["language_to_query_node"])
    graph.add_node("sql_validation_node", nodes["sql_validation_node"])
    graph.add_node("data_extraction_node", nodes["data_extraction_node"])
    graph.add_node("validation_answer_node", nodes["validation_answer_node"])

    graph.add_edge(START, "schema_detection_node")
    graph.add_edge("schema_detection_node", "language_to_query_node")
    graph.add_edge("language_to_query_node", "sql_validation_node")
    graph.add_edge("sql_validation_node", "data_extraction_node")
    graph.add_edge("data_extraction_node", "validation_answer_node")
    graph.add_edge("validation_answer_node", END)
    return graph.compile()
