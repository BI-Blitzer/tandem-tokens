"""Persist a job's results to disk and build the compact digest Claude reads.

THE TOKEN-ECONOMY CONTRACT lives here: full per-item output is written to
results/<job_id>/items/*.json and merged.jsonl and is NEVER surfaced to Claude.
Claude reads only digest.md, which is hard-capped at DIGEST_MAX_BYTES.

Layout written by write_results():
  results/<job_id>/
    items/<index>.json        one file per successful item (the value)
    merged.jsonl              one line per successful item: {"index","source","value"}
    failed_items.jsonl        one line per failed item: {"index","source","error","attempts"}
    digest.md                 the bounded human/Claude-facing summary
    run.json                  machine-readable stats (for status/harvest)
"""

from __future__ import annotations

import json
from pathlib import Path

from .jobspec import JobSpec, Mode
from .ledger import append_entry as append_ledger_entry
from .pricing import Pricing, estimate_all
from .worker import RunResult

DIGEST_MAX_BYTES = 8192
_SAMPLE_COUNT = 3
_ERROR_TRUNC = 200


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _savings_lines(run: RunResult, pricing: Pricing) -> list[str]:
    """The token-accounting block: what the map phase would have cost on each
    cloud model. Estimate only — see pricing.py for the tokenizer caveat."""
    pt, ct = run.total_prompt_tokens, run.total_completion_tokens
    rows = estimate_all(pricing, pt, ct)
    lines = [
        "## estimated cloud cost avoided (map phase)",
        f"local tokens processed: {pt} in / {ct} out — ran on local GPU for $0.",
        "had this run on a cloud model instead (approx; tokenizers differ):",
    ]
    for r in rows:
        flag = "  [verify price]" if r.verify else ""
        lines.append(f"- {r.label}: ~${r.usd:.4f}{flag}")
    lines.append(
        "note: this is the map-phase spend avoided; a Claude reduce/clean pass "
        "over merged.jsonl still costs real tokens."
    )
    return lines


def _gentest_lines(run: RunResult) -> list[str]:
    """Leaderboard (top survivors, scored) + per-iteration trace for gen_test."""
    lines = ["## leaderboard (top candidates by score)"]
    if not run.results:
        lines.append("- (no candidates survived)")
    for r in run.results:  # already sorted best-first by the engine
        payload = json.dumps(r.value, ensure_ascii=False)
        if len(payload) > 300:
            payload = payload[:300] + "…"
        mark = "PASS" if r.ok else "    "
        note = f"  ({r.verdict_notes})" if r.verdict_notes else ""
        lines.append(f"- [{r.score:.3f} {mark}] {payload}{note}")
    lines.append("")
    lines.append("## iterations")
    for it in run.iterations:
        lines.append(
            f"- iter {it['iteration']}: {it['candidates']} candidates, "
            f"best {it['best_score']}, global_best {it['global_best']}"
        )
    return lines


def build_digest_md(job_id: str, spec: JobSpec, run: RunResult, results_dir: Path,
                    pricing: Pricing | None = None) -> str:
    """Build the digest markdown, guaranteed <= DIGEST_MAX_BYTES bytes."""
    ok = [r for r in run.results if r.ok]
    bad = [r for r in run.results if not r.ok]
    lines: list[str] = []
    a = lines.append
    a(f"# digest: {spec.name}  (job {job_id})")
    a("")
    a(f"- mode: {spec.mode.value}   model: {run.model}")
    a(f"- items: {len(run.results)}   ok: {run.ok_count}   failed: {run.fail_count}"
      f"   success_rate: {run.success_rate:.1%}")
    a(f"- wall: {run.wall_seconds}s   throughput: {run.tokens_per_second:.1f} tok/s"
      f"   tokens: {run.total_prompt_tokens} in / {run.total_completion_tokens} out")
    if run.cached_count:
        a(f"- cache: {run.cached_count}/{len(run.results)} items served from cache "
          f"(0 new tokens)")
    if run.escalated_count:
        a(f"- escalated: {run.escalated_count} item(s) recovered on a stronger model "
          f"(via retry.escalate_to)")
    a(f"- artifacts: {results_dir.as_posix()}  (merged.jsonl, items/, failed_items.jsonl)")
    a("")
    is_embed = spec.mode == Mode.embed
    # Chat pricing doesn't apply to embeddings — skip the savings block for embed.
    if pricing is not None and not is_embed:
        lines.extend(_savings_lines(run, pricing))
        a("")

    if is_embed:  # embed mode: report vectors as dim/count, never dump them
        dims = {r.value.get("dim") for r in ok if isinstance(r.value, dict)}
        a("## embeddings")
        a(f"- vectors: {run.ok_count}/{len(run.results)}   dimension: "
          f"{sorted(d for d in dims if d) or '-'}")
        a(f"- stored in {(results_dir / 'merged.jsonl').as_posix()} "
          f"(one {{index, source, value:{{dim, embedding}}}} per line)")
        a("")
        if bad:
            a(f"## failures ({len(bad)})")
            for r in bad:
                err = r.error if len(r.error) <= _ERROR_TRUNC else r.error[:_ERROR_TRUNC] + "…"
                a(f"- [{r.index}] {r.source}: {err}")
        else:
            a("## failures: none")
        text = "\n".join(lines) + "\n"
        return text if len(text.encode("utf-8")) <= DIGEST_MAX_BYTES else (
            text.encode("utf-8")[: DIGEST_MAX_BYTES - 40].decode("utf-8", "ignore")
            + "\n… (truncated)\n")

    if run.stages:  # pipeline mode: stage funnel, then the map-shaped body
        a("## pipeline stages")
        for st in run.stages:
            a(f"- {st['stage']} ({st['model']}): in {st['in']}, ok {st['ok']}, "
              f"failed {st['failed']}, continued {st['continued']}")
        a("")

    if run.iterations:  # gen_test mode: leaderboard + iteration trace
        lines.extend(_gentest_lines(run))
    else:               # map / pipeline: samples + failures
        a("## sample outputs")
        for r in ok[:_SAMPLE_COUNT]:
            payload = json.dumps(r.value, ensure_ascii=False)
            if len(payload) > 400:
                payload = payload[:400] + "…"
            a(f"- [{r.index}] {r.source}: {payload}")
        if not ok:
            a("- (none succeeded)")
        a("")
        if bad:
            a(f"## failures ({len(bad)}) — need attention")
            for r in bad:
                err = r.error if len(r.error) <= _ERROR_TRUNC else r.error[:_ERROR_TRUNC] + "…"
                a(f"- [{r.index}] {r.source} (after {r.attempts} attempts): {err}")
        else:
            a("## failures: none")

    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) <= DIGEST_MAX_BYTES:
        return text
    # Over budget: truncate the failures section, keep the header/stats/samples.
    keep = text.encode("utf-8")[: DIGEST_MAX_BYTES - 64]
    truncated = keep.decode("utf-8", errors="ignore")
    return truncated + "\n… (digest truncated to 8 KB; see failed_items.jsonl)\n"


def write_results(job_id: str, spec: JobSpec, run: RunResult, root: Path,
                  pricing: Pricing | None = None) -> Path:
    """Write all artifacts under root/<job_id>/. Returns that directory."""
    results_dir = root / job_id
    items_dir = results_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    merged: list[dict] = []
    for r in run.results:
        if r.ok:
            (items_dir / f"{r.index}.json").write_text(
                json.dumps(r.value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            merged.append({"index": r.index, "source": r.source, "value": r.value})
    _write_jsonl(results_dir / "merged.jsonl", merged)

    failed = [
        {"index": r.index, "source": r.source, "error": r.error, "attempts": r.attempts}
        for r in run.results
        if not r.ok
    ]
    _write_jsonl(results_dir / "failed_items.jsonl", failed)

    run_json = {
        "job_id": job_id,
        "name": spec.name,
        "mode": spec.mode.value,
        "model": run.model,
        "items": len(run.results),
        "ok": run.ok_count,
        "failed": run.fail_count,
        "cached": run.cached_count,
        "escalated": run.escalated_count,
        "success_rate": run.success_rate,
        "wall_seconds": run.wall_seconds,
        "tokens_per_second": run.tokens_per_second,
        "prompt_tokens": run.total_prompt_tokens,
        "completion_tokens": run.total_completion_tokens,
    }
    if pricing is not None:
        run_json["estimated_cloud_cost_usd"] = {
            r.model: r.usd
            for r in estimate_all(
                pricing, run.total_prompt_tokens, run.total_completion_tokens
            )
        }
    (results_dir / "run.json").write_text(
        json.dumps(run_json, indent=2), encoding="utf-8"
    )

    digest = build_digest_md(job_id, spec, run, results_dir, pricing)
    (results_dir / "digest.md").write_text(digest, encoding="utf-8")

    # Append this job to the cumulative savings ledger (the running scoreboard).
    append_ledger_entry(root, {k: run_json[k] for k in (
        "job_id", "name", "mode", "model", "items", "ok", "failed",
        "cached", "escalated", "wall_seconds", "prompt_tokens", "completion_tokens",
    )} | ({"estimated_cloud_cost_usd": run_json["estimated_cloud_cost_usd"]}
          if "estimated_cloud_cost_usd" in run_json else {}))
    return results_dir
