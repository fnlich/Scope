# Grading benchmark

The benchmark measures the Docker grader with synthetic submissions. It does
not use leased challenges, hidden tests, miner responses, wallets, or network
services.

Run it on validator-class hardware from the repository root:

```bash
. .venv/bin/activate
python scripts/benchmark_grading.py \
  --submissions 123 \
  --tests 10 \
  --concurrency 16 \
  --repeats 5
```

Repeat with 62 and 123 submissions and with each concurrency supported by the
host. Record the instance type, CPU count, memory, storage type, architecture,
Docker version, image digest, and the complete JSON output.

The report includes per-challenge wall time, submission latency, submissions
per second, test processes per second, and failure counts. Monitor host CPU,
memory, throttling, and disk use separately during each run.

The synthetic workload measures sandbox overhead rather than model-generated
program complexity. A deployment decision should also use a run over a
representative local archive. Test cases and submitted code should remain on
the benchmark host.
