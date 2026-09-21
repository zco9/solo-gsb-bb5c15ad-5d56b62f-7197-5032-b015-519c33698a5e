"""Consecutive-threshold alerting: rule CRUD, streak semantics, gap interplay."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "test.db"),
        window_size=50,
        gap_timeout_seconds=0.25,
        sweep_enabled=False,  # deterministic: gaps are declared lazily on access
    )
    with TestClient(create_app(settings)) as c:
        yield c


def submit(client, device, seqs, value=40.0, field="temp"):
    """Submit records whose payload is {field: value} (default violates temp>30)."""
    return submit_payloads(client, device, [(s, {field: value}) for s in seqs])


def submit_payloads(client, device, entries):
    resp = client.post(
        "/api/v1/batches",
        json={
            "device_id": device,
            "records": [{"seq": s, "payload": p} for s, p in entries],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def create_rule(client, device="dev-a", field="temp", op="gt", threshold=30.0, consecutive=3):
    resp = client.post(
        f"/api/v1/devices/{device}/alert-rules",
        json={"field": field, "op": op, "threshold": threshold, "consecutive": consecutive},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def alerts(client, device="dev-a", **params):
    resp = client.get(f"/api/v1/devices/{device}/alerts", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()["alerts"]


def rule_state(client, device="dev-a", rule_index=0):
    body = client.get(f"/api/v1/devices/{device}/alert-rules").json()
    return body["rules"][rule_index]["state"]


def spans(alert_list):
    return [(a["start_seq"], a["end_seq"]) for a in alert_list]


# --------------------------------------------------------------------- #
# rule creation & listing
# --------------------------------------------------------------------- #

def test_create_and_list_rules(client):
    rule = create_rule(client, field="temp", op="gt", threshold=30.0, consecutive=3)
    assert rule["rule_id"] == 1
    assert rule["device_id"] == "dev-a"
    assert rule["state"] == {"last_seq": None, "streak": 0, "alerted": False}

    create_rule(client, field="hum", op="lt", threshold=10.0, consecutive=2)
    body = client.get("/api/v1/devices/dev-a/alert-rules").json()
    assert body["device_id"] == "dev-a"
    assert [(r["field"], r["op"], r["threshold"], r["consecutive"]) for r in body["rules"]] == [
        ("temp", "gt", 30.0, 3),
        ("hum", "lt", 10.0, 2),
    ]
    # rules are device-scoped
    assert client.get("/api/v1/devices/other/alert-rules").json()["rules"] == []


def test_rule_validation(client):
    for bad in (
        {"field": "", "op": "gt", "threshold": 1.0, "consecutive": 1},
        {"field": "temp", "op": "gte", "threshold": 1.0, "consecutive": 1},
        {"field": "temp", "op": "gt", "threshold": 1.0, "consecutive": 0},
        {"field": "temp", "op": "gt", "threshold": "hot", "consecutive": 1},
        {"field": "temp", "op": "gt", "threshold": 1.0, "consecutive": "three"},
    ):
        resp = client.post("/api/v1/devices/dev-a/alert-rules", json=bad)
        assert resp.status_code == 422, bad


def test_non_finite_threshold_rejected(client):
    resp = client.post(
        "/api/v1/devices/dev-a/alert-rules",
        content='{"field": "temp", "op": "gt", "threshold": NaN, "consecutive": 2}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_rule_for_unknown_device_allowed(client):
    # Devices are created lazily by their first batch; rules may be
    # pre-provisioned and evaluate records committed from then on.
    create_rule(client, device="dev-future", consecutive=3)
    submit(client, "dev-future", [1, 2, 3])
    assert len(alerts(client, "dev-future")) == 1


# --------------------------------------------------------------------- #
# streak semantics
# --------------------------------------------------------------------- #

def test_alert_fires_once_per_violating_run(client):
    create_rule(client, consecutive=3)
    submit(client, "dev-a", [1, 2, 3, 4, 5])  # all violate (40 > 30)
    got = alerts(client)
    assert len(got) == 1
    alert = got[0]
    assert (alert["start_seq"], alert["end_seq"]) == (1, 3)
    assert alert["trigger_value"] == 40.0
    assert alert["op"] == "gt" and alert["threshold"] == 30.0
    assert alert["field"] == "temp" and alert["consecutive"] == 3
    # the run continues but does not fire again
    state = rule_state(client)
    assert state == {"last_seq": 5, "streak": 5, "alerted": True}


def test_retrigger_only_after_recovery(client):
    create_rule(client, consecutive=2)
    submit(client, "dev-a", [1, 2])          # alert fires at 2
    submit(client, "dev-a", [3], value=10.0)  # recovers (10 <= 30)
    assert rule_state(client) == {"last_seq": 3, "streak": 0, "alerted": False}
    submit(client, "dev-a", [4, 5])          # new run -> second alert at 5
    assert spans(alerts(client)) == [(1, 2), (4, 5)]


def test_lt_operator(client):
    create_rule(client, op="lt", threshold=5.0, consecutive=2)
    submit(client, "dev-a", [1], value=3.0)
    submit(client, "dev-a", [2], value=7.0)   # not below threshold
    submit(client, "dev-a", [3, 4], value=1.0)
    assert spans(alerts(client)) == [(3, 4)]


def test_missing_or_non_numeric_field_breaks_streak(client):
    create_rule(client, consecutive=3)
    submit(client, "dev-a", [1, 2])  # streak 2
    submit_payloads(client, "dev-a", [
        (3, {"other": 1}),          # field missing
        (4, {"temp": "hot"}),       # non-numeric
        (5, {"temp": None}),        # null
        (6, {"temp": True}),        # bool is not a measurement
    ])
    # NaN can never violate (strict JSON clients cannot emit it, so post raw)
    resp = client.post(
        "/api/v1/batches",
        content='{"device_id": "dev-a", "records": [{"seq": 7, "payload": {"temp": NaN}}]}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200, resp.text
    submit(client, "dev-a", [8, 9, 10])
    assert spans(alerts(client)) == [(8, 10)]


def test_consecutive_one_alerts_on_first_violation(client):
    create_rule(client, consecutive=1)
    submit(client, "dev-a", [1], value=10.0)  # normal, no alert
    submit(client, "dev-a", [2])
    assert spans(alerts(client)) == [(2, 2)]


def test_multiple_rules_evaluate_independently(client):
    create_rule(client, field="temp", op="gt", threshold=30.0, consecutive=2)
    create_rule(client, field="hum", op="lt", threshold=20.0, consecutive=2)
    submit_payloads(client, "dev-a", [
        (1, {"temp": 40, "hum": 50}),
        (2, {"temp": 40, "hum": 10}),
        (3, {"temp": 40, "hum": 10}),
    ])
    by_rule = {}
    for a in alerts(client):
        by_rule.setdefault(a["rule_id"], []).append((a["start_seq"], a["end_seq"]))
    assert by_rule[1] == [(1, 2)]  # temp violated from seq 1
    assert by_rule[2] == [(2, 3)]  # hum violated from seq 2


def test_rule_starts_from_creation_no_history_backfill(client):
    submit(client, "dev-a", [1, 2, 3])  # committed before the rule exists
    create_rule(client, consecutive=2)
    assert alerts(client) == []  # no retroactive evaluation
    submit(client, "dev-a", [4, 5])
    assert spans(alerts(client)) == [(4, 5)]


# --------------------------------------------------------------------- #
# commit-path semantics
# --------------------------------------------------------------------- #

def test_buffered_records_are_not_evaluated_until_committed(client):
    create_rule(client, consecutive=3)
    submit(client, "dev-a", [1, 3, 4, 5])  # 1 committed; 3,4,5 buffered
    assert alerts(client) == []
    assert rule_state(client)["streak"] == 1
    submit(client, "dev-a", [2])  # commits 2, drains 3,4,5 in seq order
    assert spans(alerts(client)) == [(1, 3)]  # fired when 3 drained
    assert rule_state(client)["streak"] == 5


def test_drain_evaluates_each_record_not_just_the_batch(client):
    create_rule(client, consecutive=2)
    submit_payloads(client, "dev-a", [
        (1, {"temp": 40}),
        (3, {"temp": 40}),
        (4, {"temp": 10}),  # breaks the run once drained
        (5, {"temp": 40}),
    ])
    submit(client, "dev-a", [2], value=40.0)
    # evaluation order 1..5 -> streaks 1, 2 (alert@2), 3, 0, 1
    assert spans(alerts(client)) == [(1, 2)]
    assert rule_state(client) == {"last_seq": 5, "streak": 1, "alerted": False}


def test_duplicate_batch_does_not_double_count(client):
    create_rule(client, consecutive=3)
    submit(client, "dev-a", [1, 2, 3])
    body = submit(client, "dev-a", [1, 2, 3])  # full batch retry
    assert all(r["status"] == "duplicate" for r in body["results"])
    assert len(alerts(client)) == 1
    # the retry neither extends nor resets the streak
    assert rule_state(client) == {"last_seq": 3, "streak": 3, "alerted": True}
    submit(client, "dev-a", [4])  # still the same run: no second alert
    assert len(alerts(client)) == 1


def test_stale_and_out_of_window_records_are_not_evaluated(client):
    create_rule(client, consecutive=2)
    submit(client, "dev-a", [100, 101])  # bootstrap at 100; both violate -> alert
    assert spans(alerts(client)) == [(100, 101)]
    body = submit(client, "dev-a", [40])  # stale: below the window, never recorded
    assert body["results"][0]["status"] == "stale"
    body = submit(client, "dev-a", [10_000])  # out_of_window
    assert body["results"][0]["status"] == "out_of_window"
    # neither record touched the rule: no new alert, streak untouched
    assert spans(alerts(client)) == [(100, 101)]
    assert rule_state(client) == {"last_seq": 101, "streak": 2, "alerted": True}


# --------------------------------------------------------------------- #
# gap (断档) interplay
# --------------------------------------------------------------------- #

def test_declared_gap_breaks_streak_and_backfill_does_not_recompute(client):
    create_rule(client, consecutive=3)
    submit(client, "dev-a", [1, 2])  # streak 2
    submit(client, "dev-a", [5])     # buffered behind the pending gap 3..4
    time.sleep(0.3)                  # let the gap expire
    win = client.get("/api/v1/devices/dev-a/window").json()  # lazy 断档 declaration
    assert [(g["start_seq"], g["end_seq"]) for g in win["gaps"]] == [(3, 4)]
    # the declared gap broke the run; 5 drained to committed starts a fresh one
    assert rule_state(client)["streak"] == 1
    assert alerts(client) == []

    # 断档补录: replay fills 3 and 4 with violating values -> must NOT count
    client.post("/api/v1/devices/dev-a/gaps/1/replay")
    body = submit(client, "dev-a", [3, 4])
    assert [r["status"] for r in body["results"]] == ["gap_filled", "gap_filled"]
    assert rule_state(client)["streak"] == 1

    submit(client, "dev-a", [6, 7])  # streak 2, then 3 -> alert
    assert spans(alerts(client)) == [(5, 7)]


def test_recovery_after_gap_rearms_rule(client):
    create_rule(client, consecutive=2)
    submit(client, "dev-a", [1, 2])  # alert fires at 2
    submit(client, "dev-a", [4])     # buffered behind pending gap 3
    time.sleep(0.3)
    client.get("/api/v1/devices/dev-a/window")  # declares gap [3,3], drains 4
    assert len(alerts(client)) == 1  # the gap broke the alerted run, no new alert
    submit(client, "dev-a", [5])     # streak 2 after re-arm -> fires again
    assert spans(alerts(client)) == [(1, 2), (4, 5)]


# --------------------------------------------------------------------- #
# alert queries
# --------------------------------------------------------------------- #

def test_alerts_listing_and_rule_filter(client):
    rule = create_rule(client, consecutive=2)
    submit(client, "dev-a", [1, 2])
    body = client.get("/api/v1/devices/dev-a/alerts").json()
    assert body["device_id"] == "dev-a"
    assert len(body["alerts"]) == 1
    alert = body["alerts"][0]
    assert alert["alert_id"] == 1
    assert alert["rule_id"] == rule["rule_id"]
    assert alert["created_at"] > 0

    assert alerts(client, rule_id=rule["rule_id"]) == body["alerts"]
    assert alerts(client, rule_id=999) == []
    assert alerts(client, "unknown-device") == []
