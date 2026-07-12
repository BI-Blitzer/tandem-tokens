# Real data drop zone

Put production exports here (gitignored via `data/*` except `hybrid_campaign/`).
Job specs under `jobs/real_*.job.json` point at these paths.

## Expected files

| Path | Use case | Format |
|------|----------|--------|
| `contacts.csv` | UC1 clean | columns: `id,name,email,phone` (header required) |
| `support_tickets.jsonl` | UC2 pipeline | one JSON object per line: `{"id": int, "text": "..."}` |
| `docs.jsonl` | UC3 summarize | `{"id","domain","title","body"}` per line |

## Quick start

```pwsh
# 1) Drop files into data/real/
# 2) Submit (daemon must be running)
ollamaq submit jobs/real_uc1_contacts.job.json --job-id real-uc1
ollamaq submit jobs/real_uc2_support.job.json --job-id real-uc2
ollamaq submit jobs/real_uc3_docs.job.json --job-id real-uc3

# 3) Harvest
ollamaq digest real-uc1
py scripts/normalize_contacts_from_source.py data/real/contacts.csv results/real-uc1/merged.jsonl
```

If a file is missing, `ollamaq run` fails fast with a clear path error — that is intentional.
