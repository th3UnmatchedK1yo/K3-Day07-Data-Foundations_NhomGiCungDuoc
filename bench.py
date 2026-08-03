"""Benchmark for the UIT RAG pipeline: baseline vs. high_accuracy retrieval strategies.

Retrieval is benchmarked against the 5 core gold queries in
``benchmark/gold_queries.json`` plus the refusal-focused query in
``benchmark/diagnostic_queries.json``. Relevance is judged by
(procedure_slug correctness) AND (gold-keyword coverage over a threshold),
never by a fixed chunk_id, since baseline and high_accuracy use different
chunking strategies.

Examples::

    python bench.py --compare-strategies --skip-generation
    python bench.py --compare-strategies
    python bench.py --strategy high_accuracy --embedding-model BAAI/bge-m3
    python bench.py --compare-strategies --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Every run writes ``benchmark/results/latest.json`` and
``benchmark/results/latest.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT_DIR / "data" / "uit" / "quy-trinh-danh-cho-sinh-vien.md"
GOLD_QUERIES_PATH = ROOT_DIR / "benchmark" / "gold_queries.json"
DIAGNOSTIC_QUERIES_PATH = ROOT_DIR / "benchmark" / "diagnostic_queries.json"
RESULTS_DIR = ROOT_DIR / "benchmark" / "results"

KEYWORD_RELEVANCE_THRESHOLD = 0.5  # fraction of gold_keywords a chunk must cover to count as "relevant"
CITATION_RE = re.compile(r"\[uit_student_procedures/([^/\]]+)/([^\]]+)\]")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_for_match(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def is_relevant_chunk(chunk: dict[str, Any], expected_procedure: str, gold_keywords: list[str]) -> bool:
    """Relevance = correct procedure_slug AND sufficient gold-keyword coverage."""
    metadata = chunk.get("metadata", {})
    if metadata.get("procedure_slug") != expected_procedure:
        return False
    if not gold_keywords:
        return True
    haystack = _normalize_for_match(chunk.get("raw_text") or chunk.get("content", ""))
    hits = sum(1 for keyword in gold_keywords if _normalize_for_match(keyword) in haystack)
    return (hits / len(gold_keywords)) >= KEYWORD_RELEVANCE_THRESHOLD


def keyword_coverage(text: str, gold_keywords: list[str]) -> float:
    if not gold_keywords:
        return 1.0
    haystack = _normalize_for_match(text)
    hits = sum(1 for keyword in gold_keywords if _normalize_for_match(keyword) in haystack)
    return hits / len(gold_keywords)


def extract_citations(answer_text: str) -> list[tuple[str, str]]:
    return CITATION_RE.findall(answer_text)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def compute_retrieval_metrics(per_query_results: list[dict[str, Any]]) -> dict[str, Any]:
    hits_at_1, recall_at_3, recall_at_5, reciprocal_ranks, procedure_correct, latencies = [], [], [], [], [], []

    for entry in per_query_results:
        flags: list[bool] = entry["relevance_flags"]
        hits_at_1.append(1.0 if flags[:1] and flags[0] else 0.0)
        recall_at_3.append(1.0 if any(flags[:3]) else 0.0)
        recall_at_5.append(1.0 if any(flags[:5]) else 0.0)
        reciprocal_rank = 0.0
        for rank, relevant in enumerate(flags[:5], start=1):
            if relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)
        procedure_correct.append(1.0 if entry["predicted_procedure"] == entry["expected_procedure"] else 0.0)
        latencies.append(entry["retrieval_latency_ms"])

    return {
        "hit_at_1": _mean(hits_at_1),
        "recall_at_3": _mean(recall_at_3),
        "recall_at_5": _mean(recall_at_5),
        "mrr_at_5": _mean(reciprocal_ranks),
        "procedure_accuracy": _mean(procedure_correct),
        "mean_retrieval_latency_ms": _mean(latencies),
        "p95_retrieval_latency_ms": _p95(latencies),
    }


def run_llm_judge(deepseek_client, question: str, gold_keywords: list[str], answer: str) -> dict[str, Any]:
    """Optional, supplementary-only LLM-as-judge score (never replaces retrieval metrics)."""
    judge_prompt = (
        "Bạn là giám khảo chấm điểm câu trả lời RAG. Cho điểm từ 1 (kém) đến 5 (xuất sắc) "
        "dựa trên mức độ câu trả lời bao phủ đúng các ý bắt buộc sau, và có trích dẫn hợp lý hay không. "
        "CHỈ trả về một số nguyên từ 1 đến 5, không giải thích thêm.\n\n"
        f"Câu hỏi: {question}\n"
        f"Các ý bắt buộc phải có: {', '.join(gold_keywords)}\n"
        f"Câu trả lời cần chấm: {answer}"
    )
    try:
        raw_score = deepseek_client.generate(judge_prompt, context="(không áp dụng context cho việc chấm điểm)")
        match = re.search(r"[1-5]", raw_score)
        score = int(match.group()) if match else None
        return {"score": score, "raw": raw_score.strip()}
    except Exception as exc:  # noqa: BLE001 - judge failures must not crash the benchmark
        logger.warning("LLM judge call failed: %s", exc)
        return {"score": None, "raw": None, "error": str(exc)}


def run_strategy(
    strategy: str,
    source_path: Path,
    gold_queries: list[dict[str, Any]],
    diagnostic_queries: list[dict[str, Any]],
    embedding_model: Optional[str],
    skip_generation: bool,
    use_reranker: bool,
    use_llm_judge: bool,
) -> dict[str, Any]:
    from src.deepseek_client import DeepSeekClient, DeepSeekConfigurationError, REFUSAL_MESSAGE
    from src.rag_pipeline import UITRAGPipeline

    logger.info("=== Building index for strategy=%s ===", strategy)
    pipeline = UITRAGPipeline.for_strategy(strategy, embedding_model=embedding_model, use_reranker=use_reranker)
    build_start = time.perf_counter()
    pipeline.build_index(str(source_path))
    build_ms = (time.perf_counter() - build_start) * 1000

    deepseek_client = None
    deepseek_error: Optional[str] = None
    if not skip_generation:
        try:
            deepseek_client = DeepSeekClient()
        except DeepSeekConfigurationError as exc:
            deepseek_error = str(exc)
            logger.warning("Generation disabled for this run: %s", exc)

    per_query_results: list[dict[str, Any]] = []
    for gold in gold_queries:
        retrieval_start = time.perf_counter()
        detailed = pipeline.retrieve_detailed(gold["question"], top_k=5)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        primary_chunks = detailed["primary_chunks"]
        relevance_flags = [
            is_relevant_chunk(chunk, gold["expected_procedure"], gold["gold_keywords"]) for chunk in primary_chunks
        ]

        entry: dict[str, Any] = {
            "id": gold["id"],
            "question": gold["question"],
            "expected_procedure": gold["expected_procedure"],
            "predicted_procedure": detailed["procedure_slug"],
            "procedure_confidence": detailed["procedure_confidence"],
            "metadata_filter_applied": detailed["metadata_filter_applied"],
            "reranker_used": detailed["reranker_used"],
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "relevance_flags": relevance_flags,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "procedure_slug": chunk["metadata"].get("procedure_slug"),
                    "score": chunk["score"],
                    "source": chunk["source"],
                    "relevant": relevance_flags[index],
                    "raw_text_preview": chunk["raw_text"][:220],
                }
                for index, chunk in enumerate(primary_chunks)
            ],
        }

        if deepseek_client is not None:
            try:
                answer_result = pipeline.answer(gold["question"], top_k=5)
                answer_text = answer_result["answer"]
                citations = extract_citations(answer_text)
                valid_chunk_ids = {chunk["chunk_id"] for chunk in answer_result["retrieved_chunks"]}
                unsupported = [c for c in citations if c[1] not in valid_chunk_ids]
                entry["answer"] = answer_text
                entry["citations"] = answer_result["citations"]
                entry["generation_latency_ms"] = answer_result["latency_ms"]["generation"]
                entry["answer_metrics"] = {
                    "gold_keyword_coverage": keyword_coverage(answer_text, gold["gold_keywords"]),
                    "citation_present": len(citations) > 0,
                    "citation_valid_count": len(citations) - len(unsupported),
                    "unsupported_citation_count": len(unsupported),
                }
                if use_llm_judge:
                    entry["llm_judge"] = run_llm_judge(
                        deepseek_client, gold["question"], gold["gold_keywords"], answer_text
                    )
            except DeepSeekConfigurationError as exc:
                entry["answer"] = None
                entry["generation_error"] = str(exc)
        elif deepseek_error:
            entry["generation_error"] = deepseek_error

        per_query_results.append(entry)

    diagnostic_results: list[dict[str, Any]] = []
    for diag in diagnostic_queries:
        retrieval_start = time.perf_counter()
        detailed = pipeline.retrieve_detailed(diag["question"], top_k=5)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        entry = {
            "id": diag["id"],
            "question": diag["question"],
            "predicted_procedure": detailed["procedure_slug"],
            "procedure_confidence": detailed["procedure_confidence"],
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "top_chunk_scores": [chunk["score"] for chunk in detailed["primary_chunks"][:5]],
        }
        if deepseek_client is not None:
            try:
                answer_result = pipeline.answer(diag["question"], top_k=5)
                answer_text = answer_result["answer"]
                entry["answer"] = answer_text
                entry["refusal_correct"] = REFUSAL_MESSAGE.strip() in answer_text
            except DeepSeekConfigurationError as exc:
                entry["answer"] = None
                entry["generation_error"] = str(exc)
        elif deepseek_error:
            entry["generation_error"] = deepseek_error
        diagnostic_results.append(entry)

    retrieval_metrics = compute_retrieval_metrics(per_query_results)

    answer_metrics_summary: dict[str, Any] = {}
    if deepseek_client is not None:
        scored = [q for q in per_query_results if "answer_metrics" in q]
        refusals = [d for d in diagnostic_results if "refusal_correct" in d]
        answer_metrics_summary = {
            "mean_gold_keyword_coverage": _mean([q["answer_metrics"]["gold_keyword_coverage"] for q in scored]),
            "citation_present_rate": _mean([1.0 if q["answer_metrics"]["citation_present"] else 0.0 for q in scored]),
            "total_unsupported_citations": sum(q["answer_metrics"]["unsupported_citation_count"] for q in scored),
            "mean_generation_latency_ms": _mean([q["generation_latency_ms"] for q in scored]),
            "diagnostic_refusal_correct_rate": _mean([1.0 if d["refusal_correct"] else 0.0 for d in refusals]),
        }

    reranker_active = bool(pipeline._retriever and pipeline._retriever.reranker_active)  # noqa: SLF001

    return {
        "strategy": strategy,
        "config": {
            "chunker": pipeline._chunker_kind,  # noqa: SLF001
            "use_bm25": pipeline._use_bm25,  # noqa: SLF001
            "use_metadata_filter": pipeline._use_metadata_filter,  # noqa: SLF001
            "use_reranker_requested": use_reranker,
            "reranker_active": reranker_active,
            "use_adjacent_expansion": pipeline._use_adjacent_expansion,  # noqa: SLF001
            "top_k": 5,
        },
        "embedding_model": pipeline.embedding_model,
        "deepseek_model": deepseek_client.model if deepseek_client is not None else None,
        "generation_skipped_reason": None if deepseek_client is not None else (deepseek_error or "skipped by --skip-generation"),
        "index_build_ms": round(build_ms, 2),
        "num_chunks": len(pipeline.chunks),
        "gold_query_results": per_query_results,
        "diagnostic_query_results": diagnostic_results,
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics_summary,
    }


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# UIT RAG Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: {report['timestamp']}")
    lines.append(f"- Python: {report['python_version']}")
    lines.append(f"- Source document: `{report['source_document']}`")
    lines.append(f"- LLM judge used (supplementary only): {report['llm_judge_used']}")
    lines.append("")

    strategies = report["strategies"]

    lines.append("## Bảng so sánh Retrieval Metrics")
    lines.append("")
    lines.append("| Metric | " + " | ".join(strategies.keys()) + " |")
    lines.append("|---|" + "---|" * len(strategies))
    metric_rows = [
        ("Hit@1", "hit_at_1", _fmt_pct),
        ("Recall@3", "recall_at_3", _fmt_pct),
        ("Recall@5", "recall_at_5", _fmt_pct),
        ("MRR@5", "mrr_at_5", lambda v: f"{v:.3f}"),
        ("Procedure accuracy", "procedure_accuracy", _fmt_pct),
        ("Mean retrieval latency", "mean_retrieval_latency_ms", _fmt_ms),
        ("P95 retrieval latency", "p95_retrieval_latency_ms", _fmt_ms),
    ]
    for label, key, formatter in metric_rows:
        cells = [formatter(strategies[name]["retrieval_metrics"][key]) for name in strategies]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Bảng so sánh Answer Metrics (nếu có gọi DeepSeek)")
    lines.append("")
    any_answer_metrics = any(strategies[name]["answer_metrics"] for name in strategies)
    if not any_answer_metrics:
        lines.append(
            "_Không có answer metrics -- generation đã bị bỏ qua "
            "(--skip-generation hoặc thiếu DEEPSEEK_API_KEY)._"
        )
    else:
        lines.append("| Metric | " + " | ".join(strategies.keys()) + " |")
        lines.append("|---|" + "---|" * len(strategies))
        answer_rows = [
            ("Mean gold keyword coverage", "mean_gold_keyword_coverage", _fmt_pct),
            ("Citation present rate", "citation_present_rate", _fmt_pct),
            ("Total unsupported citations", "total_unsupported_citations", lambda v: str(v)),
            ("Mean generation latency", "mean_generation_latency_ms", _fmt_ms),
            ("Diagnostic refusal correct rate", "diagnostic_refusal_correct_rate", _fmt_pct),
        ]
        for label, key, formatter in answer_rows:
            cells = []
            for name in strategies:
                metrics = strategies[name]["answer_metrics"]
                cells.append(formatter(metrics[key]) if metrics and metrics.get(key) is not None else "n/a")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Cấu hình mỗi strategy")
    lines.append("")
    for name, data in strategies.items():
        lines.append(f"### {name}")
        lines.append(f"- Embedding model thực tế: `{data['embedding_model']}`")
        lines.append(f"- DeepSeek model: `{data['deepseek_model']}`" if data["deepseek_model"] else "- DeepSeek model: (không gọi)")
        lines.append(f"- Chunker: `{data['config']['chunker']}`")
        lines.append(f"- BM25: {data['config']['use_bm25']}, Metadata filter: {data['config']['use_metadata_filter']}")
        lines.append(
            f"- Reranker yêu cầu: {data['config']['use_reranker_requested']}, "
            f"reranker thực sự hoạt động: {data['config']['reranker_active']}"
        )
        lines.append(f"- Adjacent expansion: {data['config']['use_adjacent_expansion']}")
        lines.append(f"- Số chunks trong index: {data['num_chunks']} (build in {data['index_build_ms']:.1f} ms)")
        if data["generation_skipped_reason"]:
            lines.append(f"- Generation bị bỏ qua: {data['generation_skipped_reason']}")
        lines.append("")

    lines.append("## Kết quả từng câu hỏi (5 core queries)")
    lines.append("")
    for gold_index in range(len(next(iter(strategies.values()))["gold_query_results"])):
        question = strategies[next(iter(strategies))]["gold_query_results"][gold_index]["question"]
        expected = strategies[next(iter(strategies))]["gold_query_results"][gold_index]["expected_procedure"]
        lines.append(f"### Q{gold_index + 1}: {question}")
        lines.append(f"- Expected procedure: `{expected}`")
        for name, data in strategies.items():
            entry = data["gold_query_results"][gold_index]
            top1 = entry["retrieved_chunks"][0] if entry["retrieved_chunks"] else None
            lines.append(
                f"  - **{name}**: predicted_procedure=`{entry['predicted_procedure']}` "
                f"(confidence={entry['procedure_confidence']:.2f}, filter_applied={entry['metadata_filter_applied']}); "
                f"top-1 chunk=`{top1['chunk_id'] if top1 else 'none'}` "
                f"(relevant={top1['relevant'] if top1 else False})"
            )
            if "answer_metrics" in entry:
                lines.append(
                    f"    - answer: gold_keyword_coverage={_fmt_pct(entry['answer_metrics']['gold_keyword_coverage'])}, "
                    f"citations={len(entry.get('citations', []))}, "
                    f"unsupported_citations={entry['answer_metrics']['unsupported_citation_count']}"
                )
        lines.append("")

    lines.append("## Diagnostic query (không tính vào 5 core queries)")
    lines.append("")
    for name, data in strategies.items():
        for diag in data["diagnostic_query_results"]:
            lines.append(f"- **{name}** — `{diag['question']}`")
            lines.append(
                f"  - predicted_procedure=`{diag['predicted_procedure']}` "
                f"(confidence={diag['procedure_confidence']:.2f}), top chunk scores={diag['top_chunk_scores']}"
            )
            if "refusal_correct" in diag:
                lines.append(f"  - refusal_correct={diag['refusal_correct']}")
                lines.append(f"  - answer: {diag['answer']}")
    lines.append("")

    lines.append("## Failure cases & nguyên nhân có thể")
    lines.append("")
    failure_lines: list[str] = []
    for gold_index in range(len(next(iter(strategies.values()))["gold_query_results"])):
        for name, data in strategies.items():
            entry = data["gold_query_results"][gold_index]
            if not any(entry["relevance_flags"]):
                probable_cause = "metadata filter loại đúng procedure" if (
                    entry["metadata_filter_applied"] and entry["predicted_procedure"] != entry["expected_procedure"]
                ) else "chunk không bao phủ đủ gold keywords hoặc embedding không đủ mạnh để xếp hạng đúng chunk"
                failure_lines.append(
                    f"- **{name} / Q{gold_index + 1}** (`{entry['question'][:60]}...`): "
                    f"không có chunk relevant trong top-5. Nguyên nhân có thể: {probable_cause}."
                )
    if failure_lines:
        lines.extend(failure_lines)
    else:
        lines.append("_Không phát hiện failure case (mọi strategy đều có ít nhất 1 chunk relevant trong top-5 cho mọi câu hỏi)._")
    lines.append("")

    lines.append("## Metadata filter: giúp ích hay gây hại?")
    lines.append("")
    if "high_accuracy" in strategies:
        ha = strategies["high_accuracy"]
        filter_applied_queries = [q for q in ha["gold_query_results"] if q["metadata_filter_applied"]]
        filter_helped = [q for q in filter_applied_queries if any(q["relevance_flags"])]
        filter_hurt = [q for q in filter_applied_queries if not any(q["relevance_flags"])]
        lines.append(
            f"- Metadata filter được áp dụng cho {len(filter_applied_queries)}/{len(ha['gold_query_results'])} "
            f"core queries trong strategy `high_accuracy`."
        )
        lines.append(f"- Trong số đó: {len(filter_helped)} câu vẫn có chunk relevant, {len(filter_hurt)} câu KHÔNG có chunk relevant.")
        if filter_hurt:
            lines.append(
                "- Với các câu bị ảnh hưởng xấu, nguyên nhân nhiều khả năng là confidence threshold "
                "chưa đủ chặt hoặc procedure được suy ra sai, khiến filter loại mất đúng chunk."
            )
    else:
        lines.append("_Chỉ chạy một strategy nên không thể đánh giá riêng ảnh hưởng của metadata filter trong lần chạy này._")
    lines.append("")

    lines.append("## Kết luận")
    lines.append("")
    if len(strategies) >= 2 and "baseline" in strategies and "high_accuracy" in strategies:
        baseline_recall5 = strategies["baseline"]["retrieval_metrics"]["recall_at_5"]
        ha_recall5 = strategies["high_accuracy"]["retrieval_metrics"]["recall_at_5"]
        baseline_mrr = strategies["baseline"]["retrieval_metrics"]["mrr_at_5"]
        ha_mrr = strategies["high_accuracy"]["retrieval_metrics"]["mrr_at_5"]
        if ha_recall5 > baseline_recall5 or (ha_recall5 == baseline_recall5 and ha_mrr > baseline_mrr):
            lines.append(
                f"Trên bộ 5 core queries này, `high_accuracy` cho Recall@5={_fmt_pct(ha_recall5)} và "
                f"MRR@5={ha_mrr:.3f}, cao hơn hoặc bằng `baseline` (Recall@5={_fmt_pct(baseline_recall5)}, "
                f"MRR@5={baseline_mrr:.3f}). high_accuracy có xu hướng tốt hơn trên corpus này."
            )
        elif ha_recall5 < baseline_recall5:
            lines.append(
                f"Trên bộ 5 core queries này, `baseline` (Recall@5={_fmt_pct(baseline_recall5)}) thực tế "
                f"KHÔNG thua `high_accuracy` (Recall@5={_fmt_pct(ha_recall5)}) -- benchmark không chứng minh "
                "high_accuracy tốt hơn trên corpus nhỏ này; cần thêm câu hỏi để kết luận chắc chắn hơn."
            )
        else:
            lines.append(
                "Hai strategy cho kết quả retrieval tương đương trên bộ 5 core queries này; "
                "corpus/số câu hỏi còn nhỏ nên chưa đủ để khẳng định strategy nào vượt trội."
            )
    else:
        lines.append("Chỉ một strategy được chạy trong lần benchmark này nên không có so sánh baseline vs. high_accuracy.")
    lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the UIT RAG pipeline (baseline vs. high_accuracy retrieval strategies)."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Path to the UIT markdown source file")
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--compare-strategies", action="store_true", help="Run both baseline and high_accuracy with the SAME embedding model"
    )
    strategy_group.add_argument("--strategy", choices=["baseline", "high_accuracy"], help="Run a single strategy")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Force a specific embedding model for ALL strategies being run "
            "(keeps the embedding model constant so only the retrieval strategy varies). "
            "If omitted: baseline uses the MiniLM fallback model, high_accuracy uses BAAI/bge-m3 "
            "(or its fallback) -- this is a SEPARATE experiment about embedding-model choice, not "
            "mixed with the strategy comparison."
        ),
    )
    parser.add_argument("--reranker", action="store_true", help="Enable the optional local reranker for high_accuracy")
    parser.add_argument("--skip-generation", action="store_true", help="Skip DeepSeek calls (retrieval-only benchmark)")
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Additionally run a DeepSeek LLM-as-judge score (SUPPLEMENTARY ONLY, never replaces retrieval metrics)",
    )
    return parser


def _ensure_utf8_console() -> None:
    """Best-effort: avoid UnicodeEncodeError when printing Vietnamese text on
    Windows consoles whose default codepage is not UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _load_dotenv_if_available() -> None:
    """Load ``.env`` (DEEPSEEK_API_KEY, EMBEDDING_MODEL, ...) if python-dotenv is
    installed, mirroring ``main.py``'s ``load_dotenv(override=False)`` behaviour.
    Never overrides variables already exported in the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_console()
    _load_dotenv_if_available()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: source document not found: {source_path}", file=sys.stderr)
        return 1

    try:
        gold_queries = _load_json(GOLD_QUERIES_PATH)["queries"]
        diagnostic_queries = _load_json(DIAGNOSTIC_QUERIES_PATH)["queries"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error loading benchmark queries: {exc}", file=sys.stderr)
        return 1

    strategies_to_run = ["baseline", "high_accuracy"] if args.compare_strategies else [args.strategy or "high_accuracy"]

    if args.llm_judge and args.skip_generation:
        print("Error: --llm-judge requires generation; cannot combine with --skip-generation.", file=sys.stderr)
        return 1

    strategies: dict[str, Any] = {}
    try:
        for strategy in strategies_to_run:
            strategies[strategy] = run_strategy(
                strategy=strategy,
                source_path=source_path,
                gold_queries=gold_queries,
                diagnostic_queries=diagnostic_queries,
                embedding_model=args.embedding_model,
                skip_generation=args.skip_generation,
                use_reranker=args.reranker,
                use_llm_judge=args.llm_judge,
            )
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary: report and exit non-zero
        logger.exception("Benchmark run failed")
        print(f"Error: benchmark run failed: {exc}", file=sys.stderr)
        return 1

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "source_document": str(source_path),
        "llm_judge_used": bool(args.llm_judge),
        "strategies": strategies,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "latest.json"
    md_path = RESULTS_DIR / "latest.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for name, data in strategies.items():
        metrics = data["retrieval_metrics"]
        print(
            f"[{name}] Hit@1={metrics['hit_at_1']:.2f} Recall@5={metrics['recall_at_5']:.2f} "
            f"MRR@5={metrics['mrr_at_5']:.3f} ProcAcc={metrics['procedure_accuracy']:.2f} "
            f"embedding_model={data['embedding_model']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
