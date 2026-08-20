"""Prompt templates for the Evaluator subsystem."""

from __future__ import annotations


def build_evaluation_prompt(
    question: str,
    answer: str,
    retrieved_context: str,
    evaluation_history: str | None = None,
) -> str:
    """Build the evaluation prompt sent to the LLM.

    The prompt asks the LLM to act as an answer quality evaluator and
    return a structured JSON assessment covering accuracy, relevance,
    completeness, consistency, grounding, and unsupported claims.
    """
    history_section = ""

    if evaluation_history and evaluation_history.strip():
        history_section = f"""
Previous evaluation history for this question:
{evaluation_history}

Use this history to avoid repeating previously identified issues and
to track whether earlier feedback has been addressed.
"""

    return f"""You are an expert answer quality evaluator for a RAG system.

Your task is to evaluate the generated answer against the retrieved context
and the original user question.

{history_section}
## Input

**Question:**
{question}

**Generated Answer:**
{answer}

**Retrieved Context:**
{retrieved_context}

## Evaluation Criteria

Rate each criterion from 0 to 100:

1. **accuracy** - Are the factual claims in the answer correct given the context?
2. **relevance** - Does the answer directly address the question asked?
3. **completeness** - Does the answer cover all important aspects of the question?
4. **consistency** - Are there any internal contradictions in the answer?
5. **grounding** - Is the answer supported by the retrieved context?
6. **unsupported_claims** - Are there claims not backed by the context? (higher = fewer unsupported claims)

Also assess:
- **overall_quality** - General quality of the answer (0-100)

## Decision Rules

- If overall_quality >= 75 AND grounding >= 70 AND accuracy >= 70: decision = "ACCEPT"
- Otherwise: decision = "IMPROVE"

## Output Format

Return ONLY a valid JSON object with this exact structure:

{{
    "decision": "ACCEPT" or "IMPROVE",
    "score": <overall_quality integer>,
    "criteria": {{
        "accuracy": <0-100>,
        "relevance": <0-100>,
        "completeness": <0-100>,
        "consistency": <0-100>,
        "grounding": <0-100>,
        "unsupported_claims": <0-100>,
        "overall_quality": <0-100>
    }},
    "feedback": [
        {{
            "criterion": "<criterion name>",
            "severity": "high" or "medium" or "low",
            "issue": "<description of the issue>",
            "suggestion": "<actionable suggestion for improvement>"
        }}
    ]
}}

If the answer is acceptable with no issues, return an empty feedback array.

Return ONLY the JSON object. No other text."""
