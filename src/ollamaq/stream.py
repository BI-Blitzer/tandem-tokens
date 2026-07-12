"""Streaming execution for map/embed over very large inputs.

Where the normal path materializes every chunk and every result in memory, this
path pulls chunks lazily (chunker.iter_chunks), processes them in bounded batches,
and writes each result straight to merged.jsonl / failed_items.jsonl. Peak memory
is one batch + running stats, so a multi-GB CSV streams through instead of loading
whole. Opt in with `ollamaq run --stream` (map and embed only).
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .cache import Cache
from .chunker import iter_chunks
from .client import OllamaClient
from .config import Config
from .embed import _embed_one
from .jobspec import JobSpec, Mode
from .ledger import append_entry as append_ledger_entry
from .pricing import Pricing, estimate_all
from .worker import ItemResult, _run_item

_SAMPLE_CAP = 3
_FAILURE_CAP = 100     # bound failures held for the digest; all still hit failed_items.jsonl


def _batched(it: Iterable, n: int) -> Iterator[list]:
    batch: list = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


async def run_stream(config: Config, spec: JobSpec, results_dir: Path,
                     pricing: Optional[Pricing], batch_size: int = 0) -> dict:
    """Stream a map/embed job to disk. Returns a compact stats dict for the CLI."""
    model = config.resolve_model(spec.model, spec.task_tag)
    max_parallel = config.models[model].max_parallel
    if batch_size <= 0:
        batch_size = max_parallel * 8
    is_embed = spec.mode == Mode.embed

    from .energy import EnergyMeter
    from .worker import _resolve_escalate

    cache = None
    escalate_model = escalate_num_ctx = escalate_cloud = None
    if not is_embed:
        if spec.cache:
            try:
                cap = int(os.environ.get("OLLAMAQ_CACHE_MAX_ENTRIES", "10000"))
            except ValueError:
                cap = 10000
            cache = Cache(os.environ.get("OLLAMAQ_CACHE", "cache"), max_entries=cap)
        escalate_model, escalate_num_ctx, escalate_cloud = _resolve_escalate(
            config, spec, model
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    st = {"ok": 0, "fail": 0, "items": 0, "cached": 0, "escalated": 0,
          "ptok": 0, "ctok": 0, "wall": 0.0, "samples": [], "failures": [], "dims": set(),
          "energy_wh": None, "avg_power_w": None, "peak_power_w": None}
    sem = asyncio.Semaphore(max_parallel)
    meter = EnergyMeter().start()
    start = time.perf_counter()

    cloud_client = None
    grok_cm = None
    if escalate_cloud:
        settings = config.grok_settings_or_none()
        if settings is None:
            from .errors import OllamaqError
            raise OllamaqError(
                f"escalate target {escalate_cloud!r} needs cloud.grok.enabled "
                "and XAI_API_KEY or GROK_API_KEY"
            )
        from .cloud import GrokClient
        grok_cm = GrokClient(settings)
        cloud_client = grok_cm

    merged_f = (results_dir / "merged.jsonl").open("w", encoding="utf-8")
    failed_f = (results_dir / "failed_items.jsonl").open("w", encoding="utf-8")
    try:
        async def _consume(client: OllamaClient) -> None:
            async def _run(w) -> ItemResult:
                async with sem:
                    if is_embed:
                        return await _embed_one(client, model, w.index, w.source,
                                                w.text, spec.retry.transient_retries)
                    return await _run_item(
                        client, model, spec, w.index, w.source, w.text, cache,
                        escalate_model, escalate_num_ctx, list(w.images) or None,
                        escalate_cloud=escalate_cloud, cloud_client=cloud_client,
                        config=config,
                    )

            for batch in _batched(iter_chunks(spec.input), batch_size):
                meter.tick()
                results = await asyncio.gather(*[_run(w) for w in batch])
                for r in results:
                    st["items"] += 1
                    st["ptok"] += r.prompt_tokens
                    st["ctok"] += r.completion_tokens
                    if r.cached:
                        st["cached"] += 1
                    if r.escalated:
                        st["escalated"] += 1
                    if r.ok:
                        st["ok"] += 1
                        merged_f.write(json.dumps(
                            {"index": r.index, "source": r.source, "value": r.value},
                            ensure_ascii=False) + "\n")
                        if is_embed and isinstance(r.value, dict):
                            st["dims"].add(r.value.get("dim"))
                        elif len(st["samples"]) < _SAMPLE_CAP:
                            st["samples"].append((r.index, r.source, r.value))
                    else:
                        st["fail"] += 1
                        failed_f.write(json.dumps(
                            {"index": r.index, "source": r.source,
                             "error": r.error, "attempts": r.attempts},
                            ensure_ascii=False) + "\n")
                        if len(st["failures"]) < _FAILURE_CAP:
                            st["failures"].append((r.index, r.source, r.error))

        if grok_cm is not None:
            async with OllamaClient(config.ollama) as client, grok_cm:
                await _consume(client)
        else:
            async with OllamaClient(config.ollama) as client:
                await _consume(client)
    finally:
        merged_f.close()
        failed_f.close()

    st["wall"] = round(time.perf_counter() - start, 3)
    energy = meter.stop()
    if energy.available:
        st["energy_wh"] = energy.energy_wh
        st["avg_power_w"] = energy.avg_power_w
        st["peak_power_w"] = energy.peak_power_w
    _write_stream_digest(spec, model, st, results_dir, pricing)
    return {"ok": st["ok"], "fail": st["fail"], "items": st["items"], "model": model}


def _write_stream_digest(spec: JobSpec, model: str, st: dict, results_dir: Path,
                         pricing: Optional[Pricing]) -> None:
    is_embed = spec.mode == Mode.embed
    L: list[str] = [f"# digest: {spec.name}  ({results_dir.name}) [streamed]", "",
                    f"- mode: {spec.mode.value}   model: {model}",
                    f"- items: {st['items']}   ok: {st['ok']}   failed: {st['fail']}",
                    f"- tokens: {st['ptok']} in / {st['ctok']} out"]
    if st["cached"]:
        L.append(f"- cache: {st['cached']}/{st['items']} served from cache")
    if st["escalated"]:
        L.append(f"- escalated: {st['escalated']} item(s)")
    if st.get("energy_wh") is not None:
        L.append(
            f"- energy: ~{st['energy_wh']:.4f} Wh "
            f"(avg {st.get('avg_power_w', 0):.0f} W, peak {st.get('peak_power_w', 0):.0f} W)"
        )
        try:
            from .energy_cost import cost_for_wh, load_energy_rates
            rates = load_energy_rates()
            if rates is not None:
                el = cost_for_wh(rates, float(st["energy_wh"]))
                L.append(f"- energy cost: ~${el.usd:.6f} @ ${el.usd_per_kwh:.4f}/kWh")
                if el.co2_g is not None:
                    L.append(f"- CO2e: ~{el.co2_g:.3f} g")
        except Exception:
            pass
    L.append(f"- artifacts: {results_dir.as_posix()}  (merged.jsonl, failed_items.jsonl)")
    L.append("")
    if pricing is not None and not is_embed:
        rows = estimate_all(pricing, st["ptok"], st["ctok"])
        L.append("## estimated cloud cost avoided (map phase)")
        L.append(f"local tokens: {st['ptok']} in / {st['ctok']} out - ran for $0")
        for r in rows:
            L.append(f"- {r.label}: ~${r.usd:.4f}" + ("  [verify price]" if r.verify else ""))
        L.append("")
    if is_embed:
        L.append("## embeddings")
        L.append(f"- vectors: {st['ok']}/{st['items']}   dimension: "
                 f"{sorted(d for d in st['dims'] if d) or '-'}")
    else:
        L.append("## sample outputs")
        for idx, src, val in st["samples"]:
            payload = json.dumps(val, ensure_ascii=False)
            if len(payload) > 300:
                payload = payload[:300] + "..."
            L.append(f"- [{idx}] {src}: {payload}")
        if not st["samples"]:
            L.append("- (none)")
    L.append("")
    if st["failures"]:
        L.append(f"## failures ({st['fail']}, first {len(st['failures'])})")
        for idx, src, err in st["failures"]:
            L.append(f"- [{idx}] {src}: {err[:150]}")
    else:
        L.append("## failures: none")

    text = "\n".join(L) + "\n"
    if len(text.encode("utf-8")) > 8192:
        text = text.encode("utf-8")[:8100].decode("utf-8", "ignore") + "\n... (truncated)\n"
    (results_dir / "digest.md").write_text(text, encoding="utf-8")

    run_json: dict[str, Any] = {
        "job_id": results_dir.name, "name": spec.name, "mode": spec.mode.value,
        "model": model, "host": platform.node(), "input": spec.input.describe(),
        "items": st["items"], "ok": st["ok"], "failed": st["fail"],
        "cached": st["cached"], "escalated": st["escalated"], "streamed": True,
        "success_rate": st["ok"] / st["items"] if st["items"] else 0.0,
        "wall_seconds": st["wall"],
        "prompt_tokens": st["ptok"], "completion_tokens": st["ctok"],
    }
    if st.get("energy_wh") is not None:
        run_json["energy_wh"] = round(float(st["energy_wh"]), 4)
        run_json["avg_power_w"] = round(float(st.get("avg_power_w") or 0), 2)
        run_json["peak_power_w"] = round(float(st.get("peak_power_w") or 0), 2)
        try:
            from .energy_cost import cost_for_wh, load_energy_rates
            rates = load_energy_rates()
            if rates is not None:
                el = cost_for_wh(rates, float(st["energy_wh"]))
                run_json["energy_usd"] = round(el.usd, 6)
                run_json["usd_per_kwh"] = el.usd_per_kwh
                if el.co2_g is not None:
                    run_json["co2_g"] = round(el.co2_g, 3)
        except Exception:
            pass
    if pricing is not None and not is_embed:
        run_json["estimated_cloud_cost_usd"] = {
            r.model: r.usd for r in estimate_all(pricing, st["ptok"], st["ctok"])}
    (results_dir / "run.json").write_text(json.dumps(run_json, indent=2), encoding="utf-8")
    ledger_keys = (
        "job_id", "name", "mode", "model", "host", "input", "items", "ok", "failed",
        "cached", "escalated", "wall_seconds", "prompt_tokens", "completion_tokens",
        "energy_wh", "avg_power_w", "peak_power_w", "energy_usd", "usd_per_kwh", "co2_g",
    )
    append_ledger_entry(results_dir.parent, {
        k: run_json[k] for k in ledger_keys if k in run_json
    } | ({"estimated_cloud_cost_usd": run_json["estimated_cloud_cost_usd"]}
         if "estimated_cloud_cost_usd" in run_json else {}))
