"""Locust scenario for E4 panel (a) — POST /api/v1/segment/{type} 202 latency.

What we are measuring: sidecar overhead only. The sidecar is invoked with
no env vars (`TIMEWORX_API_BASE_URL`, `TIMEWORX_API_KEY`, `GCS_BUCKET`
unset) so `dependencies.build_default_state` selects the stub Timeworx
client + stub staging service. The 202 latency therefore reflects:
auth pass-through, JSON parse, `map_legacy`, correlation-store insert,
and the BackgroundTasks enqueue — NOT any real AI or storage I/O. State
that in the figure caption so readers don't compare apples to oranges.

Body shape: legacy descriptor (`{rawDataUri, targetClass}`) — the
shortest valid body that exercises the production path. Using an
AI-Ready body would also exercise `map_ai_ready`'s heavier
`_extract_agrovoc_themes` walk, which is what we measure in
panel (b) — keep the two panels measuring distinct slices.

Run from the sidecar root (sidecar already serving on :8000):

    locust -f eval/locustfile.py --headless -u 50 -r 50 -t 60s \\
        --host http://127.0.0.1:8000 \\
        --csv=eval/results/e4_latency

For the per-concurrency table the paper reports, run three times with
different `-u`/`-r` pairs and different `--csv` prefixes:

    locust -f eval/locustfile.py ... -u 1  -r 1  --csv=eval/results/e4_latency_c1
    locust -f eval/locustfile.py ... -u 10 -r 10 --csv=eval/results/e4_latency_c10
    locust -f eval/locustfile.py ... -u 50 -r 50 --csv=eval/results/e4_latency_c50

`run_e4_perf.py` then reads `e4_latency*_stats.csv` and tags each row
with the concurrency suffix.
"""
from __future__ import annotations

import uuid

try:
    from locust import HttpUser, between, task
except ImportError as e:  # noqa: BLE001
    raise SystemExit(
        f"locust not installed ({e}); run `pip install -r requirements-dev.txt`"
    )


_LEGACY_BODY_TEMPLATE = {
    "rawDataUri": "s3://eval-bucket/locust/{uid}/",
    "targetClass": "maize plant",
}


class BrokerSubmitUser(HttpUser):
    """Fires `POST /api/v1/segment/instance` with a legacy descriptor.

    `wait_time = between(0, 0)` so concurrency `-u N` actually maps to
    `N` in-flight requests rather than `N` users pacing themselves with
    think-time. The brief's c ∈ {1, 10, 50} reading depends on that.
    """

    wait_time = between(0, 0)

    @task
    def submit_legacy(self) -> None:
        # Use a fresh URI per request so the sidecar's dedup logic, if any,
        # doesn't artificially flatten the curve.
        uid = uuid.uuid4().hex[:12]
        body = {
            "rawDataUri": _LEGACY_BODY_TEMPLATE["rawDataUri"].format(uid=uid),
            "targetClass": _LEGACY_BODY_TEMPLATE["targetClass"],
        }
        with self.client.post(
            "/api/v1/segment/instance",
            json=body,
            name="POST /api/v1/segment/instance (legacy)",
            catch_response=True,
        ) as response:
            if response.status_code == 202 and response.headers.get("Location"):
                response.success()
            else:
                response.failure(
                    f"expected 202 + Location header; got {response.status_code} "
                    f"location={response.headers.get('Location')!r}"
                )
