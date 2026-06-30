# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyen Manh Hieu — 2A202600887

**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (6.55ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (0.04ms P95 trong offline rule-precheck; NeMo LLM đo riêng khi deploy)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 5.90 | 6.55 | 6.55 | <10ms |
| Input rule precheck | 0.03 | 0.04 | 0.04 | <300ms |
| RAG Pipeline | N/A (evaluated separately) | N/A | N/A | <2000ms |
| NeMo Output Rail | Chưa đo online | Chưa đo online | Chưa đo online | <300ms |
| **Total input guard** | 5.94 | **6.58** | 6.58 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No

**Comment:** Số đo trên là Presidio + deterministic input precheck vì NeMo/LangChain hiện không tương thích Python 3.14 của máy chạy. Gate production phải đo lại NeMo LLM trên image Python 3.11 trước khi merge.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: RAGAS_MODE=auto python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: python -m pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # production target ≥ 18/20 (90%); lab minimum 15/20

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| RAGAS answer relevancy | < 0.60 | Review prompts and failed samples |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS-compatible avg_score (50q) | 0.6669 (lexical fallback run) |
| Worst metric | context_precision (47/50 câu) |
| Dominant failure distribution | factual theo raw count; adversarial theo failure rate/avg score |
| Cohen's κ | 0.8000 |
| Adversarial pass rate | 20/20 (100%) |
| Guard P95 latency | 6.58ms (input guard offline) |

---

## Nhận xét & Cải tiến

Lab đã triển khai đủ ba lớp eval/guard và có fallback để CI không phụ thuộc dịch vụ ngoài. Lần chạy hiện tại cho thấy Presidio P95 6.87ms, adversarial pass 100%, nhưng context precision chỉ 0.3014 toàn tập. Ưu tiên tiếp theo là metadata filter cho policy version, cross-encoder reranking và query decomposition cho multi-hop. Trước production cần chạy lại RAGAS và NeMo bằng LLM thật trên Python 3.11, lưu baseline riêng và không trộn số liệu fallback với số liệu online.
