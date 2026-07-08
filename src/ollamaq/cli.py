"""ollamaq command-line interface.

Phase 0 ships three commands:
  ollamaq doctor            - check config + Ollama reachability + model coverage
  ollamaq models            - print the routing table and installed-model status
  ollamaq bench [--model M] - benchmark models, write config/bench.json

Later phases add: submit, run, daemon, status, digest, cancel (see PLAN.md section 4).
Those are registered here as stubs that exit 2 with a clear "Phase N" message so
the surface is discoverable but never silently no-ops.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .bench import run_bench, write_report
from .chunker import chunk_input
from .client import OllamaClient
from .config import Config, load_config, normalize_model_name
from .daemon import run_daemon
from .dashboard import render, snapshot
from .digest import write_results
from .errors import OllamaqError
from .eta import bench_lookup, project
from .execute import execute_spec
from .gpu import detect_gpu, recommend as build_recommendation
from .jobspec import Mode, load_jobspec
from .ledger import read_entries as read_ledger, summarize as summarize_ledger
from .pricing import load_pricing
from .queue import JobQueue, State
from .stream import run_stream
from .worker import run_job

app = typer.Typer(
    add_completion=False,
    help="Local Ollama offload harness for Claude Code.",
    no_args_is_help=True,
)

DEFAULT_BENCH_PATH = Path("config") / "bench.json"
DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_QUEUE_ROOT = Path("queue")


def _load_or_exit(config_path: Optional[Path]) -> Config:
    try:
        return load_config(config_path)
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


def _run_streaming(cfg: Config, spec, job_id, results_root: Path, batch: int,
                   min_success: float) -> None:
    """--stream path: bounded-memory streaming for map/embed only."""
    if spec.mode not in (Mode.map, Mode.embed):
        typer.secho(f"error: --stream supports map and embed only "
                    f"(this is {spec.mode.value})", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        pricing = load_pricing()
    except OllamaqError:
        pricing = None
    jid = job_id or f"{spec.name}-{int(time.time())}"
    typer.echo(f"streaming '{spec.name}' as job {jid} (mode={spec.mode.value}) ...")
    try:
        res = asyncio.run(run_stream(cfg, spec, results_root / jid, pricing, batch))
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"done: {res['ok']}/{res['items']} ok, streamed to "
               f"{(results_root / jid / 'digest.md').as_posix()}")
    rate = res["ok"] / res["items"] if res["items"] else 0.0
    if spec.mode == Mode.map and rate < min_success:
        typer.secho(f"success rate {rate:.1%} below {min_success:.0%}",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)


def _run_sample(cfg: Config, spec, n: int) -> None:
    """Probe the first n map items, print a full-job ETA, and stop. Exits with a
    message for non-map modes (sampling assumes independent per-item work)."""
    if spec.mode != Mode.map:
        typer.secho(f"error: --sample is only meaningful for map jobs "
                    f"(this is {spec.mode.value})", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        work = chunk_input(spec.input)
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if not work:
        typer.secho("error: input produced 0 work items", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    n = min(n, len(work))
    typer.echo(f"sampling {n} of {len(work)} items to estimate the full job ...")
    items = [(w.index, w.source, w.text, w.images) for w in work[:n]]
    result = asyncio.run(run_job(cfg, spec, items))
    if result.fail_count:
        typer.secho(f"warning: {result.fail_count}/{n} sample items failed — "
                    "the estimate may be optimistic", fg=typer.colors.YELLOW, err=True)
    tps_fallback, load = bench_lookup(result.model)
    max_parallel = cfg.models[result.model].max_parallel
    proj = project(result, total_items=len(work), max_parallel=max_parallel,
                   tps_fallback=tps_fallback, load_seconds=load)
    typer.echo(proj.format())
    typer.echo("run without --sample to execute the full job "
               "(add \"cache\": true to reuse these sampled items).")


@app.command()
def version() -> None:
    """Print the ollamaq version and exit."""
    typer.echo(__version__)


@app.command()
def doctor(
    config_path: Optional[Path] = typer.Option(
        None, "--config", help="Path to models.yaml (else search default locations)."
    ),
) -> None:
    """Diagnose config, Ollama connectivity, and model coverage. Exit 0 iff healthy."""
    cfg = _load_or_exit(config_path)
    typer.echo(f"config: ok ({len(cfg.models)} models, {len(cfg.routing)} routes)")

    async def _probe() -> tuple[bool, list[str]]:
        async with OllamaClient(cfg.ollama) as client:
            if not await client.ping():
                return False, []
            return True, await client.list_models()

    reachable, installed = asyncio.run(_probe())
    if not reachable:
        typer.secho(
            f"ollama: UNREACHABLE at {cfg.ollama.base_url}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    typer.echo(f"ollama: ok ({cfg.ollama.base_url}, {len(installed)} models installed)")

    installed_set = {normalize_model_name(n) for n in installed}
    missing = [
        name
        for name, m in cfg.models.items()
        if m.enabled and normalize_model_name(name) not in installed_set
    ]
    if missing:
        typer.secho(
            "warning: enabled models not installed: " + ", ".join(missing),
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.secho("all enabled models are installed", fg=typer.colors.GREEN)


@app.command()
def models(
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Print the routing table and per-model settings."""
    cfg = _load_or_exit(config_path)
    typer.echo(f"defaults: model={cfg.defaults.model} temperature={cfg.defaults.temperature}")
    typer.echo("\nmodels:")
    for name, m in cfg.models.items():
        flag = "" if m.enabled else "  [disabled]"
        typer.echo(f"  {name:22} parallel={m.max_parallel} ctx={m.num_ctx}{flag}")
    typer.echo("\nrouting (task_tag -> model):")
    for tag, target in cfg.routing.items():
        typer.echo(f"  {tag:12} -> {target}")


@app.command()
def bench(
    model: Optional[list[str]] = typer.Option(
        None, "--model", "-m", help="Model(s) to benchmark; repeatable. Default: all enabled."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    out: Path = typer.Option(DEFAULT_BENCH_PATH, "--out", help="Where to write bench.json."),
) -> None:
    """Benchmark models (sequentially) and write results to bench.json."""
    cfg = _load_or_exit(config_path)
    report = asyncio.run(run_bench(cfg, list(model) if model else None))
    for r in report.results:
        if r.ok:
            typer.echo(
                f"{r.model:22} {r.tokens_per_second:7.2f} tok/s  "
                f"load={r.load_seconds:.2f}s  ({r.completion_tokens} tok)"
            )
        else:
            typer.secho(f"{r.model:22} FAILED: {r.error}", fg=typer.colors.RED, err=True)
    path = write_report(report, out)
    typer.echo(f"\nwrote {path}")
    if any(not r.ok for r in report.results):
        raise typer.Exit(code=1)


@app.command()
def run(
    jobfile: Path = typer.Argument(..., help="Path to a job spec JSON file."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    results_root: Path = typer.Option(
        DEFAULT_RESULTS_ROOT, "--results", help="Root dir for results/<job_id>/."
    ),
    job_id: Optional[str] = typer.Option(
        None, "--job-id", help="Override the job id (default: spec name + timestamp)."
    ),
    min_success: float = typer.Option(
        0.95, "--min-success", help="Exit 1 if success rate falls below this."
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="run is synchronous; use 'ollamaq submit' to enqueue."
    ),
    sample: int = typer.Option(
        0, "--sample", help="Probe the first N map items, print a full-job ETA, and stop."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Stream a huge map/embed input in bounded batches "
        "(constant memory), writing results to disk as it goes."
    ),
    batch: int = typer.Option(0, "--batch", help="Streaming batch size (0 = auto)."),
) -> None:
    """Run a job synchronously (map / gen_test / pipeline / embed) and write its digest."""
    if not wait:
        typer.secho("error: 'run' is synchronous — use 'ollamaq submit' to enqueue "
                    "a job for the daemon instead", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    cfg = _load_or_exit(config_path)
    try:
        spec = load_jobspec(jobfile)
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    if sample > 0:
        _run_sample(cfg, spec, sample)
        return

    if stream:
        _run_streaming(cfg, spec, job_id, results_root, batch, min_success)
        return

    jid = job_id or f"{spec.name}-{int(time.time())}"
    typer.echo(f"running '{spec.name}' as job {jid} (mode={spec.mode.value}) ...")

    try:
        result = execute_spec(cfg, spec)
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    try:
        pricing = load_pricing()
    except OllamaqError as exc:
        typer.secho(f"warning: {exc} (skipping savings block)",
                    fg=typer.colors.YELLOW, err=True)
        pricing = None
    results_dir = write_results(jid, spec, result, results_root, pricing)

    if result.iterations:  # gen_test: report the search outcome
        best = result.iterations[-1]["global_best"]
        typer.echo(f"done: {len(result.iterations)} iterations, {len(result.results)} "
                   f"survivors, best score {best}, {result.wall_seconds}s")
    elif result.stages:    # pipeline: report the funnel
        funnel = " -> ".join(f"{s['stage']}({s['continued']})" for s in result.stages)
        typer.echo(f"done: {len(result.stages)} stages [{funnel}], "
                   f"{result.ok_count}/{len(result.results)} ok, {result.wall_seconds}s")
    else:                  # map: report the batch outcome
        typer.echo(
            f"done: {result.ok_count}/{len(result.results)} ok "
            f"({result.success_rate:.1%}), {result.wall_seconds}s, "
            f"{result.tokens_per_second:.1f} tok/s"
        )
    typer.echo(f"digest: {(results_dir / 'digest.md').as_posix()}")
    # min_success gates map batches only; gen_test is a search, not pass/fail.
    if not result.iterations and result.success_rate < min_success:
        typer.secho(
            f"success rate {result.success_rate:.1%} below threshold {min_success:.0%}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def submit(
    jobfile: Path = typer.Argument(..., help="Path to a job spec JSON file."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    queue_root: Path = typer.Option(DEFAULT_QUEUE_ROOT, "--queue"),
    job_id: Optional[str] = typer.Option(None, "--job-id"),
) -> None:
    """Validate a job spec and enqueue it for the daemon (does not run it now)."""
    _load_or_exit(config_path)  # validate config exists before enqueuing
    try:
        spec = load_jobspec(jobfile)
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    jid = job_id or f"{spec.name}-{int(time.time())}"
    q = JobQueue(queue_root)
    try:
        q.submit(jid, spec.model_dump(mode="json", by_alias=True))
    except OllamaqError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(f"submitted {jid} to {(queue_root / 'pending').as_posix()}")


@app.command()
def daemon(
    config_path: Optional[Path] = typer.Option(None, "--config"),
    queue_root: Path = typer.Option(DEFAULT_QUEUE_ROOT, "--queue"),
    results_root: Path = typer.Option(DEFAULT_RESULTS_ROOT, "--results"),
    min_success: float = typer.Option(0.95, "--min-success"),
    poll_interval: float = typer.Option(5.0, "--poll", help="Seconds to sleep when idle."),
) -> None:
    """Drain the job queue continuously (foreground). Run ONE per machine."""
    cfg = _load_or_exit(config_path)
    try:
        pricing = load_pricing()
    except OllamaqError:
        pricing = None
    q = JobQueue(queue_root)

    def _log(m: str) -> None:
        # flush so progress is visible when stdout is redirected to a log file
        # (the documented Task Scheduler / pythonw launch).
        typer.echo(m)
        sys.stdout.flush()

    try:
        run_daemon(cfg, q, results_root, pricing, min_success, poll_interval, log=_log)
    except KeyboardInterrupt:
        typer.echo("\ndaemon interrupted; exiting")


@app.command()
def status(
    job_id: Optional[str] = typer.Argument(None, help="Show one job; omit for a summary."),
    queue_root: Path = typer.Option(DEFAULT_QUEUE_ROOT, "--queue"),
) -> None:
    """Show queue state — a counts summary, or details for one job id."""
    q = JobQueue(queue_root)
    if job_id:
        found = q.find(job_id)
        if not found:
            typer.secho(f"no such job: {job_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        st, job = found
        typer.echo(f"{job_id}: {st.value}   model={job.model or '?'}"
                   f"   error={job.error or '-'}")
        return
    for st in State:
        jobs = q.list_state(st)
        typer.echo(f"{st.value:8} {len(jobs)}")
        for j in jobs[:20]:
            typer.echo(f"    {j.job_id}")


@app.command()
def watch(
    queue_root: Path = typer.Option(DEFAULT_QUEUE_ROOT, "--queue"),
    results_root: Path = typer.Option(DEFAULT_RESULTS_ROOT, "--results"),
    interval: float = typer.Option(2.0, "--interval", help="Refresh seconds."),
) -> None:
    """Live dashboard: queue state + savings scoreboard, refreshed in place."""
    q = JobQueue(queue_root)
    try:
        while True:
            snap = snapshot(q, results_root)
            sys.stdout.write("\033[2J\033[H")  # clear + home (VT-enabled consoles)
            sys.stdout.write(render(snap))
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("\nstopped")


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="Pending job id to cancel."),
    queue_root: Path = typer.Option(DEFAULT_QUEUE_ROOT, "--queue"),
) -> None:
    """Cancel a pending job (running/finished jobs cannot be cancelled)."""
    q = JobQueue(queue_root)
    if q.cancel(job_id):
        typer.echo(f"cancelled {job_id}")
    else:
        typer.secho(f"{job_id} is not pending (already running/done, or unknown)",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)


@app.command()
def recommend(
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Recommend which configured models fit this GPU (vs. spill to CPU), using
    nvidia-smi VRAM + installed model sizes + bench.json throughput if present."""
    cfg = _load_or_exit(config_path)
    gpu = detect_gpu()

    async def _sizes() -> dict:
        async with OllamaClient(cfg.ollama) as client:
            if not await client.ping():
                return {}
            return await client.model_sizes()

    try:
        sizes = asyncio.run(_sizes())
    except OllamaqError:
        sizes = {}
    bench = None
    bp = DEFAULT_BENCH_PATH
    if bp.is_file():
        import json as _json
        try:
            bench = _json.loads(bp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            bench = None
    rec = build_recommendation(cfg, sizes, gpu, bench)
    typer.echo(rec.format())
    if bench is None:
        typer.echo("\n(tip: run `ollamaq bench` to add measured tok/s to this report)")


@app.command()
def score(
    results_root: Path = typer.Option(DEFAULT_RESULTS_ROOT, "--results"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Print the cumulative scoreboard: local tokens run for $0 and the estimated
    cloud spend avoided across every job in the ledger."""
    entries = read_ledger(results_root)
    sb = summarize_ledger(entries)
    # headline against the pricing default_compare model when available
    headline = None
    try:
        pricing = load_pricing()
        if pricing is not None:
            headline = pricing.default_compare
    except OllamaqError:
        pass
    typer.echo(sb.format(headline_model=headline))


@app.command()
def digest(
    job_id: str = typer.Argument(..., help="Job id whose digest.md to print."),
    results_root: Path = typer.Option(DEFAULT_RESULTS_ROOT, "--results"),
) -> None:
    """Print an existing job's digest.md to stdout."""
    path = results_root / job_id / "digest.md"
    if not path.is_file():
        typer.secho(f"error: no digest at {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.echo(path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    app()
