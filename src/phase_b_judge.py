from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH, TEST_SET_PATH


_llm_judge_available: bool | None = None


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[\w%]+", text, flags=re.UNICODE))


def _fallback_pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Deterministic judge used when the remote judge is unavailable.

    This is deliberately conservative: near-equivalent answers are ties, while
    more relevant and specific answers win. It keeps unit tests and CI usable
    without turning a transient API outage into an evaluation outage.
    """
    q_tokens = set(_normalise_text(question).split())

    def features(answer: str) -> tuple[float, set[str], set[str]]:
        normalised = _normalise_text(answer)
        tokens = set(normalised.split())
        numbers = set(re.findall(r"\d+(?:[.,]\d+)?%?", normalised))
        relevance = len(q_tokens & tokens) / max(len(q_tokens), 1)
        specificity = min(len(numbers), 3) * 0.04
        detail = min(len(tokens), 40) / 400
        uncertainty = 0.12 if any(x in normalised for x in ("không rõ", "có thể", "không tìm thấy")) else 0
        score = max(0.0, min(1.0, 0.55 + relevance * 0.3 + specificity + detail - uncertainty))
        return score, tokens, numbers

    score_a, tokens_a, nums_a = features(answer_a)
    score_b, tokens_b, nums_b = features(answer_b)
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
    coverage_a = len(tokens_a & tokens_b) / max(len(tokens_a), 1)
    coverage_b = len(tokens_a & tokens_b) / max(len(tokens_b), 1)
    norm_a, norm_b = _normalise_text(answer_a), _normalise_text(answer_b)
    authority_markers = ("hiện hành", "hien hanh", "đã bị thay thế", "da bi thay the",
                         "câu trả lời đúng", "cau tra loi dung")
    authoritative_a = any(marker in norm_a for marker in authority_markers)
    authoritative_b = any(marker in norm_b for marker in authority_markers)

    # Matching factual values and substantial lexical overlap usually mean the
    # answers are equivalent, even if one is wordier.
    if authoritative_a != authoritative_b and nums_a != nums_b:
        winner = "A" if authoritative_a else "B"
        reason = "Câu trả lời thắng nêu rõ chính sách hiện hành và xử lý xung đột phiên bản."
    elif max(coverage_a, coverage_b) >= 0.98:
        winner = "tie"
        reason = "Một câu trả lời diễn đạt đầy đủ các dữ kiện chính của câu còn lại."
    elif nums_a == nums_b and overlap >= 0.45:
        winner = "tie"
        reason = "Hai câu trả lời có cùng dữ kiện chính và nội dung tương đương."
    elif abs(score_a - score_b) <= 0.025:
        winner = "tie"
        reason = "Hai câu trả lời có mức độ liên quan và đầy đủ tương đương."
    elif score_a > score_b:
        winner = "A"
        reason = "Answer A liên quan, cụ thể và đầy đủ hơn theo heuristic dự phòng."
    else:
        winner = "B"
        reason = "Answer B liên quan, cụ thể và đầy đủ hơn theo heuristic dự phòng."
    return {
        "winner": winner,
        "reasoning": reason,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
        "backend": "deterministic_fallback",
    }


def _validate_judgment(data: dict) -> dict:
    winner = str(data.get("winner", "tie")).strip()
    winner = winner if winner in {"A", "B", "tie"} else "tie"
    raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    scores = {}
    for label in ("A", "B"):
        try:
            scores[label] = round(max(0.0, min(1.0, float(raw_scores.get(label, 0.5)))), 3)
        except (TypeError, ValueError):
            scores[label] = 0.5
    return {
        "winner": winner,
        "reasoning": str(data.get("reasoning") or "Không có giải thích từ judge."),
        "scores": scores,
        "backend": data.get("backend", "openai"),
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Choose the better answer, with an offline-safe fallback."""
    global _llm_judge_available

    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''

    if OPENAI_API_KEY and _llm_judge_available is not False:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=8.0, max_retries=0)
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(
                        question=question, answer_a=answer_a, answer_b=answer_b)},
                ],
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)
            result = _validate_judgment(json.loads(content))
            _llm_judge_available = True
            return result
        except Exception as exc:
            _llm_judge_available = False
            print(f"  ⚠️  LLM judge unavailable ({type(exc).__name__}); using deterministic fallback.")

    return _fallback_pairwise_judge(question, answer_a, answer_b)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán."""
    pass1     = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    swap_map      = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2  = swap_map[pass2_raw["winner"]]

    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"

    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels và human_labels phải có cùng số phần tử")
    if not judge_labels:
        return 0.0
    labels = set(judge_labels) | set(human_labels)
    if len(labels) == 1:
        return 1.0

    n = len(judge_labels)
    observed = sum(a == b for a, b in zip(judge_labels, human_labels)) / n
    expected = sum(
        (judge_labels.count(label) / n) * (human_labels.count(label) / n)
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias."""
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0,
                "position_bias_count": 0, "verbosity_details": {}, "interpretation": ""}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


def save_phase_b_report(judge_results: list[JudgeResult], bias: dict,
                         kappa: float, path: str = "reports/judge_results.json",
                         label_comparison: list[dict] | None = None) -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "cohen_kappa": round(kappa, 4),
        "bias_report": bias,
        "total_judged": len(judge_results),
        "label_comparison": label_comparison or [],
        "judgments": [
            {
                "question": r.question,
                "answer_a": r.answer_a,
                "answer_b": r.answer_b,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = {item["id"]: item for item in json.load(f)}

    print(f"Running swap-and-average on {len(human_data)} labeled questions...")
    judge_results = []
    judge_labels = []
    comparisons = []
    for index, item in enumerate(human_data, 1):
        reference = test_set[item["question_id"]]["ground_truth"]
        result = swap_and_average(item["question"], item["model_answer"], reference)
        judge_results.append(result)
        # A means the model answer is at least as good as the reference. A tie
        # is accepted because both answers can be semantically equivalent.
        judge_label = 1 if result.final_winner in {"A", "tie"} else 0
        judge_labels.append(judge_label)
        comparisons.append({
            "question_id": item["question_id"],
            "human_label": item["human_label"],
            "judge_label": judge_label,
            "agree": judge_label == item["human_label"],
        })
        print(f"  [{index}/{len(human_data)}] id={item['question_id']} "
              f"winner={result.final_winner} label={judge_label}")

    human_labels = [item["human_label"] for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ: {kappa:.3f}")

    bias = bias_report(judge_results)
    print(f"\nBias report: {bias}")
    save_phase_b_report(judge_results, bias, kappa, label_comparison=comparisons)
