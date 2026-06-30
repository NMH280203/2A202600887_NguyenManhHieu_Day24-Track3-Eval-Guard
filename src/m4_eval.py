from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re, unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts và ground_truths phải có cùng độ dài")
    if not questions:
        return {
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0,
            "per_question": [], "backend": "empty",
        }

    zeros = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }
    mode = os.getenv("RAGAS_MODE", "auto").lower()
    try:
        if mode in {"lexical", "offline"}:
            raise RuntimeError("lexical evaluation requested")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
            "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
            "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
            "per_question": per_question,
            "backend": "ragas",
        }
    except Exception as e:
        if mode not in {"lexical", "offline"}:
            print(f"  ⚠️  RAGAS evaluation failed: {e}; using lexical fallback")
        return _evaluate_lexical(questions, answers, contexts, ground_truths)


_STOPWORDS = {
    "và", "là", "của", "có", "được", "cho", "trong", "khi", "một", "những",
    "các", "theo", "với", "này", "đó", "thì", "về", "bao", "nhiêu", "gì",
    "the", "a", "an", "of", "to", "and", "is", "are", "in", "for",
}


def _tokens(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", str(text)).lower()
    return [token for token in re.findall(r"[\w%]+", normalised, flags=re.UNICODE)
            if len(token) > 1 and token not in _STOPWORDS]


def _coverage(source: str, target: str) -> float:
    """Fraction of target information tokens supported by source."""
    source_tokens, target_tokens = set(_tokens(source)), set(_tokens(target))
    if not target_tokens:
        return 1.0 if not source_tokens else 0.0
    return len(source_tokens & target_tokens) / len(target_tokens)


def _f1(left: str, right: str) -> float:
    lset, rset = set(_tokens(left)), set(_tokens(right))
    if not lset or not rset:
        return 0.0
    overlap = len(lset & rset)
    precision, recall = overlap / len(lset), overlap / len(rset)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _evaluate_lexical(questions: list[str], answers: list[str],
                      contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Dependency-free approximation used only when RAGAS/LLM is unavailable."""
    per_question = []
    for question, answer, context_list, ground_truth in zip(
        questions, answers, contexts, ground_truths
    ):
        context_list = [str(item) for item in (context_list or [])]
        joined_context = "\n".join(context_list)
        answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", answer))
        context_numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", joined_context))
        unsupported_numbers = answer_numbers - context_numbers

        faithfulness = _coverage(joined_context, answer)
        if unsupported_numbers:
            faithfulness *= max(0.0, 1.0 - 0.2 * len(unsupported_numbers))
        answer_relevancy = 0.65 * _f1(answer, ground_truth) + 0.35 * _coverage(answer, question)
        if context_list:
            relevance_scores = [max(_f1(ctx, question), _f1(ctx, ground_truth)) for ctx in context_list]
            context_precision_score = sum(relevance_scores) / len(relevance_scores)
        else:
            context_precision_score = 0.0
        context_recall_score = _coverage(joined_context, ground_truth)

        values = [faithfulness, answer_relevancy, context_precision_score, context_recall_score]
        values = [round(max(0.0, min(1.0, value)), 4) for value in values]
        per_question.append(EvalResult(
            question=question, answer=answer, contexts=context_list, ground_truth=ground_truth,
            faithfulness=values[0], answer_relevancy=values[1],
            context_precision=values[2], context_recall=values[3],
        ))

    def mean(metric: str) -> float:
        return sum(getattr(item, metric) for item in per_question) / len(per_question)

    return {
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "per_question": per_question,
        "backend": "lexical_fallback",
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    failures = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        avg_score = sum(metrics.values()) / len(metrics)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": avg_score,
            "metrics": metrics,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    failures.sort(key=lambda x: x["score"])
    return failures[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
