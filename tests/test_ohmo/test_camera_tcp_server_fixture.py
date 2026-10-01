"""Two-process smoke checks for the scratch Camera TCP server fixture."""

from __future__ import annotations

import http.client
import json
import os
import selectors
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from ohmo.camera_protocol.models import candidate_id_for
from ohmo.gateway.attachment_fingerprints import (
    PHASH_HAMMING_THRESHOLD,
    fingerprint_image_bytes,
    phash_hamming_distance,
)

OHMO_ROOT = Path(os.environ.get("OHMO_CHECKOUT", Path(__file__).parents[2])).resolve()
FIXTURE = Path(__file__).parent / "helpers" / "camera_tcp_server_fixture.py"
sys.path.insert(0, str(OHMO_ROOT / "tests" / "test_ohmo"))
from test_camera_ingress import _candidate, _multipart as source_multipart, _upload  # noqa: E402


class ServerProcess:
    def __init__(self, workspace: Path, token_file: Path, *, hold_photo: bool = False):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src:."
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(FIXTURE),
                "--workspace",
                str(workspace),
                "--token-file",
                str(token_file),
                "--ohmo-checkout",
                str(OHMO_ROOT),
                *( ["--hold-photo"] if hold_photo else [] ),
            ],
            cwd=OHMO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        ready = self.read_json(timeout=10)
        assert ready["ready"] is True
        assert ready["host"] == "127.0.0.1"
        self.port = ready["port"]

    def read_json(self, *, timeout: float = 5) -> dict:
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("fixture did not emit bounded JSON control output")
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read(2048).decode("utf-8", "replace")
            raise RuntimeError(f"fixture exited before response ({error[:120]!r})")
        value = json.loads(line)
        assert isinstance(value, dict)
        return value

    def control(self, op: str, **values) -> dict:
        self.process.stdin.write((json.dumps({"op": op, **values}) + "\n").encode())
        self.process.stdin.flush()
        return self.read_json()

    def close(self):
        if self.process.poll() is None:
            try:
                self.control("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)


def _http(server: ServerProcess, path: str, body: bytes, *, content_type: str, token: str, extra=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
    }
    headers.update(extra or {})
    connection.request("POST", path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


def _lease(server: ServerProcess, token: str) -> dict:
    status, body = _http(
        server,
        "/internal/v1/camera/session",
        b"",
        content_type="application/json",
        token=token,
    )
    assert status == 200
    return json.loads(body)


def _references(
    server: ServerProcess,
    token: str,
    candidate_id: str,
    capture_time: str,
    *,
    schema_version: int | None = None,
):
    query = {"candidate_id": candidate_id, "capture_time": capture_time}
    if schema_version is not None:
        query["schema_version"] = schema_version
    return _http(
        server,
        "/internal/v1/camera/references",
        json.dumps(query).encode(),
        content_type="application/json",
        token=token,
    )


def _multipart(root: Path, index: int, lease: dict, *, captured_at=None):
    request = _candidate(root, index=index, capture_time=captured_at)
    leased = {
        **request,
        "session_id": lease["session_id"],
        "epoch": lease["epoch"],
        "seq": lease["committed_seq"] + 1,
    }
    upload = _upload(root, leased)
    content_type, body = source_multipart(upload)
    return leased, content_type, body


def _run_scene_stage(checkout, producer_python, state_dir, server, token, stage, verdict):
    runner = Path(__file__).parent / "helpers" / "camera_telegent_pipeline_runner.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            str(producer_python), str(runner),
            "--telegent-checkout", str(checkout),
            "--state-dir", str(state_dir),
            "--port", str(server.port),
            "--token", token,
            "--phase", "initial",
            "--scenario", "scene_source_repair",
            "--stage", stage,
            "--scene-verdict", verdict,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-4000:]
    return json.loads(result.stdout)


def test_real_loopback_ingress_replay_auth_parser_reconcile_and_restart(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"s" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file)
    token = "s" * 40
    producer_root = workspace / "synthetic-producer"
    producer_root.mkdir(mode=0o700)

    try:
        lease = _lease(server, token)
        leased, content_type, body = _multipart(producer_root, 0, lease)
        path = "/internal/v1/camera/candidates"
        status, first_body = _http(
            server, path, body, content_type=content_type, token=token
        )
        assert status == 202
        first = json.loads(first_body)
        assert first["status"] == "admitted"
        assert first["delivery_semantics"] == "in_process_only"
        idle = server.control("await_idle")
        assert idle["ok"] is True
        after_first = idle["counters"]
        assert after_first["photo_calls"] == 1
        assert after_first["confirmed_photo_attempts"] == 1
        assert after_first["queued_messages"] == 1

        replay_status, replay_body = _http(
            server, path, body, content_type=content_type, token=token
        )
        assert replay_status == 202
        assert json.loads(replay_body) == first
        after_replay = server.control("counters")["counters"]
        assert after_replay["photo_calls"] == 1
        assert after_replay["journal_entries"] == 1

        before_rejections = after_replay
        unauthorized_status, _ = _http(
            server, path, body, content_type=content_type, token="wrong"
        )
        assert unauthorized_status == 401
        malformed_status, _ = _http(
            server,
            path,
            b"--malformed\r\nnot-a-part\r\n--malformed--\r\n",
            content_type="multipart/form-data; boundary=malformed",
            token=token,
        )
        assert malformed_status == 400
        after_rejections = server.control("counters")["counters"]
        for field in ("journal_entries", "attempt_states", "committed_seq", "photo_calls"):
            assert after_rejections[field] == before_rejections[field]

        lease = _lease(server, token)
        before_absent = server.control("counters")["counters"]
        _, content_type2, body2 = _multipart(producer_root, 1, lease)
        absent_status, absent_body = _http(
            server,
            "/internal/v1/camera/reconcile",
            body2,
            content_type=content_type2,
            token=token,
            extra={"X-Camera-Reconcile-Purpose": "existing_outcome_only"},
        )
        assert absent_status == 503
        assert json.loads(absent_body)["error"]["code"] == "unknown_original_outcome"
        after_absent = server.control("counters")["counters"]
        assert after_absent["journal_entries"] == 1
        assert after_absent["committed_seq"] == 1
        assert after_absent["photo_calls"] == 1
        assert after_absent["journal_sha256"] == before_absent["journal_sha256"]
        assert after_absent["journal_mtime_ns"] == before_absent["journal_mtime_ns"]

        lease = _lease(server, token)
        old_capture = datetime.now(timezone.utc) - timedelta(days=8)
        _, content_type3, body3 = _multipart(
            producer_root, 2, lease, captured_at=old_capture
        )
        retired_status, retired_body = _http(
            server,
            "/internal/v1/camera/reconcile",
            body3,
            content_type=content_type3,
            token=token,
            extra={"X-Camera-Reconcile-Purpose": "retire_ineligible"},
        )
        assert retired_status == 200
        assert json.loads(retired_body)["status"] == "retired"
        before_restart = server.control("counters")["counters"]
        assert before_restart["attempt_states"].get("retired") == 1
        assert before_restart["photo_calls"] == 1

        restarted = server.control("restart")
        assert restarted["ok"] is True
        assert restarted["journal_preserved"] is True
        server.port = restarted["port"]
        after_restart = server.control("counters")["counters"]
        assert after_restart["journal_entries"] == 2
        assert after_restart["photo_calls"] == 1
        assert after_restart["confirmed_photo_attempts"] == 1
    finally:
        server.close()


def test_stdin_eof_closes_listener_cleanly(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"t" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file)
    server.process.stdin.close()
    assert server.process.wait(timeout=5) == 0
    assert server.process.poll() == 0


def test_real_loopback_references_pending_then_native_receipt_are_read_only(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"r" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file, hold_photo=True)
    token = "r" * 40
    producer_root = workspace / "synthetic-producer"
    producer_root.mkdir(mode=0o700)
    try:
        capture = datetime.now(timezone.utc).replace(microsecond=0)
        lease = _lease(server, token)
        candidate, content_type, body = _multipart(
            producer_root, 9, lease, captured_at=capture
        )
        status, response_body = _http(
            server, "/internal/v1/camera/candidates", body,
            content_type=content_type, token=token,
        )
        assert status == 202
        admitted = json.loads(response_body)
        assert admitted["status"] == "admitted"
        _pending = server.control("counters")["counters"]
        assert _pending["photo_calls"] == 0
        before = (_pending["journal_sha256"], _pending["journal_mtime_ns"])
        query_id = _candidate(producer_root, index=99, capture_time=capture)["candidate_id"]
        status, response_body = _references(server, token, query_id, candidate["capture_time"])
        assert status == 200, response_body
        response = json.loads(response_body)
        assert response["references"] == []
        assert response["pending_candidate_ids"] == [candidate["candidate_id"]]
        after = server.control("counters")["counters"]
        assert (after["journal_sha256"], after["journal_mtime_ns"]) == before

        assert server.control("release_photos")["ok"] is True
        delivered = server.control("await_idle")["counters"]
        assert delivered["photo_calls"] == 1
        assert delivered["confirmed_photo_attempts"] == 1
        before = (delivered["journal_sha256"], delivered["journal_mtime_ns"])
        status, response_body = _references(server, token, query_id, candidate["capture_time"])
        assert status == 200
        response = json.loads(response_body)
        assert response["pending_candidate_ids"] == []
        assert len(response["references"]) == 1
        reference = response["references"][0]
        assert {key: reference[key] for key in (
            "candidate_id", "image_sha256", "capture_time", "capture_time_authority"
        )} == {
            "candidate_id": candidate["candidate_id"],
            "image_sha256": candidate["image_sha256"],
            "capture_time": capture.isoformat(),
            "capture_time_authority": candidate["capture_time_authority"],
        }
        native_id = reference["native_photo_message_id"]
        assert type(native_id) is int and native_id > 0
        after = server.control("counters")["counters"]
        assert (after["journal_sha256"], after["journal_mtime_ns"]) == before
    finally:
        server.close()


def test_explicit_telegent_pipeline_lost_ack_restart_and_old_epoch_replay(tmp_path):
    checkout_value = os.environ.get("TELEGENT_CHECKOUT")
    python_value = os.environ.get("TELEGENT_PYTHON")
    if not checkout_value or not python_value:
        pytest.skip("set TELEGENT_CHECKOUT and TELEGENT_PYTHON for the cross-repo fixture")
    checkout = Path(checkout_value).resolve(strict=True)
    producer_python = Path(python_value).absolute()
    assert checkout.is_dir() and (checkout / "telegent/health_advisor/dropbox_camera/pipeline.py").is_file()
    assert producer_python.is_file() and producer_python.name.startswith("python")

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"p" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file, hold_photo=True)
    token = "p" * 40
    producer_state = tmp_path / "producer-state"
    producer_state.mkdir(mode=0o700)
    runner = Path(__file__).parent / "helpers" / "camera_telegent_pipeline_runner.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [
            str(producer_python), str(runner),
            "--telegent-checkout", str(checkout),
            "--state-dir", str(producer_state),
            "--port", str(server.port),
            "--token", token,
            "--phase", "initial",
            "--scenario", "lost_ack_replay",
        ],
        cwd=checkout,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )

    def read_report(timeout: float = 60) -> dict:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("Telegent pipeline fixture did not emit bounded JSON")
        line = process.stdout.readline()
        if not line:
            error = process.stderr.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"Telegent fixture exited early: {error[:2000]}")
        value = json.loads(line)
        assert isinstance(value, dict)
        return value

    try:
        initial = read_report()
        assert initial["phase"] == "initial" and initial["drop_occurred"] is True
        assert initial["sequences"] == [1]
        assert initial["outcomes"] == ["submission_unknown", "submission_deferred"], (
            initial["results"],
            [(item["response_status"], item["response_body"]) for item in initial["observed"]],
        )
        first_request = initial["observed"][0]["request"]
        lost_request = initial["active_request"]
        assert first_request["seq"] == 1 and first_request == lost_request
        assert first_request["source_revision"] == "rev:tcp-b"

        counters = server.control("counters")["counters"]
        assert counters["journal_entries"] == 1
        assert counters["photo_calls"] == 0
        before = (counters["journal_sha256"], counters["journal_mtime_ns"])
        query_id = "dropbox-camera-v1-" + "f" * 64
        status, body = _references(server, token, query_id, lost_request["capture_time"])
        assert status == 200
        pending = json.loads(body)
        assert pending["references"] == []
        assert set(pending["pending_candidate_ids"]) == {
            first_request["candidate_id"]
        }
        counters = server.control("counters")["counters"]
        assert (counters["journal_sha256"], counters["journal_mtime_ns"]) == before

        assert server.control("release_photos")["ok"] is True
        delivered = server.control("await_idle")["counters"]
        assert delivered["photo_calls"] == 1
        assert delivered["confirmed_photo_attempts"] == 1
        before = (delivered["journal_sha256"], delivered["journal_mtime_ns"])
        status, body = _references(server, token, query_id, lost_request["capture_time"])
        assert status == 200
        confirmed = json.loads(body)
        assert confirmed["pending_candidate_ids"] == []
        assert [item["candidate_id"] for item in confirmed["references"]] == [
            lost_request["candidate_id"]
        ]
        assert all(type(item["native_photo_message_id"]) is int
                   and item["native_photo_message_id"] > 0 for item in confirmed["references"])
        after = server.control("counters")["counters"]
        assert (after["journal_sha256"], after["journal_mtime_ns"]) == before

        restarted = server.control("restart")
        assert restarted["journal_preserved"] is True
        server.port = restarted["port"]
        assert server.control("counters")["counters"]["committed_seq"] == 0
        process.stdin.write((json.dumps({"op": "reopen", "port": server.port}) + "\n").encode())
        process.stdin.flush()
        replay = read_report()
        assert replay["phase"] == "reopen"
        assert replay["recovery_outcome"] == "accepted"
        assert sorted(replay["outcomes"]) == sorted(["duplicate_suppression", "submission_deferred"]), (
            replay["results"],
            [(item["path"], item["purpose"], item["response_status"], item["response_body"])
             for item in replay["observed"]],
        )
        assert replay["state_committed_seq"] == 0
        assert replay["lease"]["committed_seq"] == 0
        assert len(replay["observed"]) == 2
        assert replay["observed"][0]["method"] == "POST"
        assert replay["observed"][0]["path"] == "/internal/v1/camera/reconcile"
        assert replay["observed"][0]["purpose"] == "existing_outcome_only"
        assert replay["observed"][0]["request"] == lost_request
        assert replay["observed"][1]["request"]["seq"] == 1
        assert replay["observed"][1]["path"] == "/internal/v1/camera/candidates"
        assert replay["observed"][1]["response_status"] == 409
        assert replay["active_request"]["candidate_id"] != lost_request["candidate_id"]
        assert replay["attempt_results"][lost_request["candidate_id"]]["outcome"] == "accepted"
        final = server.control("counters")["counters"]
        assert final["journal_entries"] == 1
        assert final["photo_calls"] == 1
        assert final["confirmed_photo_attempts"] == 1
        assert final["committed_seq"] == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        server.close()


def test_telegent_expired_unknown_request_retires_without_fresh_photo(tmp_path):
    checkout_value = os.environ.get("TELEGENT_CHECKOUT")
    python_value = os.environ.get("TELEGENT_PYTHON")
    if not checkout_value or not python_value:
        pytest.skip("set TELEGENT_CHECKOUT and TELEGENT_PYTHON for the cross-repo fixture")
    checkout = Path(checkout_value).resolve(strict=True)
    producer_python = Path(python_value).absolute()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"e" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file)
    producer_state = tmp_path / "producer-state"
    producer_state.mkdir(mode=0o700)
    runner = Path(__file__).parent / "helpers" / "camera_telegent_pipeline_runner.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [
            str(producer_python), str(runner),
            "--telegent-checkout", str(checkout),
            "--state-dir", str(producer_state),
            "--port", str(server.port),
            "--token", "e" * 40,
            "--phase", "initial",
            "--scenario", "negative_expiry",
        ],
        cwd=checkout,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )

    def read_report() -> dict:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(60):
                raise TimeoutError("Telegent expiry fixture did not emit bounded JSON")
        line = process.stdout.readline()
        if not line:
            error = process.stderr.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"Telegent expiry fixture exited early: {error[:1000]}")
        return json.loads(line)

    try:
        initial = read_report()
        assert initial["drop_occurred"] is True
        assert initial["outcomes"][0] == "submission_unknown"
        assert initial["observed"][0]["path"] == "/internal/v1/camera/candidates"
        assert initial["observed"][0]["response_status"] is None
        assert server.control("counters")["counters"]["journal_entries"] == 0

        clock_path = producer_state / "capture-clock.json"
        producer_clock = datetime.fromisoformat(json.loads(clock_path.read_text())["now"])
        clock_path.write_text(
            json.dumps({"now": (producer_clock + timedelta(days=8)).isoformat()}),
            encoding="utf-8",
        )
        assert server.control("advance_camera_days", days=8)["ok"] is True
        process.stdin.write((json.dumps({"op": "reopen", "port": server.port}) + "\n").encode())
        process.stdin.flush()
        retired = read_report()
        assert retired["outcomes"].count("submission_retired") == 1
        assert len(retired["observed"]) == 1
        assert retired["observed"][0]["path"] == "/internal/v1/camera/reconcile"
        assert retired["observed"][0]["purpose"] == "retire_ineligible"
        assert retired["observed"][0]["response_status"] == 200
        assert json.loads(retired["observed"][0]["response_body"])["status"] == "retired"
        consumer = server.control("counters")["counters"]
        assert consumer["journal_entries"] == 1
        assert consumer["attempt_states"] == {"retired": 1}
        assert consumer["photo_calls"] == 0
        assert consumer["confirmed_photo_attempts"] == 0
        assert consumer["committed_seq"] == 0
        assert retired["state_committed_seq"] == 0
        assert retired["attempt_results"][initial["active_request"]["candidate_id"]][
            "outcome"
        ] == "retired"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        server.close()


def test_actual_scene_producer_fails_closed_on_legacy_source_conflict(
    tmp_path,
):
    checkout_value = os.environ.get("TELEGENT_CHECKOUT")
    python_value = os.environ.get("TELEGENT_PYTHON")
    if not checkout_value or not python_value:
        pytest.skip("set TELEGENT_CHECKOUT and TELEGENT_PYTHON for the cross-repo fixture")
    checkout = Path(checkout_value).resolve(strict=True)
    producer_python = Path(python_value).absolute()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"c" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file)
    token = "c" * 40
    producer_state = tmp_path / "producer-state"
    producer_state.mkdir(mode=0o700)
    _lease(server, token)
    producer_clock = server.control("counters")["counters"]["camera_now"]
    (producer_state / "capture-clock.json").write_text(
        json.dumps({"now": producer_clock}), encoding="utf-8"
    )
    try:
        initial = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "initial", "non_food"
        )
        assert any(item["path"] == "/internal/v1/camera/candidates"
                   for item in initial["observed"]), initial
        assert [item["request"]["seq"] for item in initial["observed"]] == [1]
        confirmed = server.control("await_idle")["counters"]
        assert confirmed["photo_calls"] == confirmed["confirmed_photo_attempts"] == 1
        conflict = server.control("make_legacy_source_conflict")
        a_id = conflict["candidate_id"]
        assert conflict["ok"] is True, conflict
        confirmed_history = confirmed["attempt_history"][a_id]
        assert conflict["original_image_sha256"] == confirmed_history["image_sha256"]
        assert conflict["conflicting_image_sha256"] != conflict["original_image_sha256"]
        restarted = server.control("restart")
        assert restarted["journal_preserved"] is True
        server.port = restarted["port"]
        pre_repair = server.control("counters")["counters"]
        assert pre_repair["reference_source_records"] == 0
        before_history = pre_repair["attempt_history"][a_id]
        assert before_history["request_identity"] == confirmed_history["request_identity"]
        assert before_history["request_ack"] == confirmed_history["request_ack"]
        assert before_history["photo_id"] == confirmed_history["photo_id"]
        assert before_history["photo_delivery_confirmed"] is True
        assert before_history["image_sha256"] == conflict["conflicting_image_sha256"]
        read_before = (pre_repair["journal_sha256"], pre_repair["journal_mtime_ns"])

        distinct = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "distinct", "non_food"
        )
        assert distinct["scene_source_operations"] == 1, {
            "outcomes": distinct["outcomes"],
            "results": distinct["results"],
            "queries": [
                (item["request"], item["response_status"], item["response_body"])
                for item in distinct["reference_queries"]
            ],
            "source_operations": distinct["source_operations"],
            "observed": [
                (item["path"], item["response_status"], item["response_body"])
                for item in distinct["observed"]
            ],
        }
        assert distinct["scene_sources_reestablished"] == 0
        assert distinct["scene_reference_hold_count"] == 1
        assert distinct["scene_provider_requests"] == 0
        assert len(distinct["source_operations"]) == 1
        source_op = distinct["source_operations"][0]
        assert source_op["request"]["image_sha256"] == conflict["original_image_sha256"]
        assert source_op["image_sha256"] == conflict["original_image_sha256"]
        assert source_op["response_status"] == 409
        assert json.loads(source_op["response_body"])["error"][
            "code"
        ] == "reference_source_conflict"
        assert len(distinct["reference_queries"]) == 2
        assert all(item["response_status"] == 200 for item in distinct["reference_queries"])
        assert all(
            json.loads(item["response_body"])["coverage"] == "incomplete"
            and json.loads(item["response_body"])["unresolved_candidate_ids"] == [a_id]
            for item in distinct["reference_queries"]
        )
        assert all(
            item["path"] != "/internal/v1/camera/candidates"
            or item["request"]["source_revision"] != "rev:tcp-b"
            for item in distinct["observed"]
        )
        assert "/internal/v1/camera/session" not in distinct["operation_order"]
        after_hold = server.control("counters")["counters"]
        assert (after_hold["journal_sha256"], after_hold["journal_mtime_ns"]) == read_before
        assert after_hold["photo_calls"] == after_hold["confirmed_photo_attempts"] == 1
        assert after_hold["camera_meal_commits"] == 0
        assert after_hold["attempt_history"][a_id]["image_sha256"] == (
            conflict["conflicting_image_sha256"]
        )
        assert after_hold["attempt_history"][a_id]["request_identity"] == (
            confirmed_history["request_identity"]
        )
        assert after_hold["attempt_history"][a_id]["request_ack"] == (
            confirmed_history["request_ack"]
        )
        assert after_hold["journal_entries"] == 1
        assert after_hold["committed_seq"] == 0
    finally:
        server.close()


def test_actual_scene_producer_restores_immutable_legacy_source_before_admitting_next_photo(
    tmp_path,
):
    checkout_value = os.environ.get("TELEGENT_CHECKOUT")
    python_value = os.environ.get("TELEGENT_PYTHON")
    if not checkout_value or not python_value:
        pytest.skip("set TELEGENT_CHECKOUT and TELEGENT_PYTHON for the cross-repo fixture")
    checkout = Path(checkout_value).resolve(strict=True)
    producer_python = Path(python_value).absolute()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"v" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file, hold_photo=True)
    token = "v" * 40
    producer_state = tmp_path / "producer-state"
    producer_state.mkdir(mode=0o700)
    _lease(server, token)
    producer_clock = server.control("counters")["counters"]["camera_now"]
    (producer_state / "capture-clock.json").write_text(
        json.dumps({"now": producer_clock}), encoding="utf-8"
    )
    try:
        initial = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "initial", "non_food"
        )
        a_id = initial["target_candidate_id"]
        a_post = next(
            item for item in initial["observed"]
            if item["path"] == "/internal/v1/camera/candidates"
        )
        assert a_post["response_status"] == 202, {
            "status": a_post["response_status"],
            "body": a_post["response_body"],
        }
        a_ack = json.loads(a_post["response_body"])
        assert a_ack["candidate_id"] == a_id and a_ack["ack_seq"] == 1
        assert server.control("release_photos")["ok"] is True
        a_delivered = server.control("await_idle")["counters"]
        assert a_delivered["photo_calls"] == a_delivered["confirmed_photo_attempts"] == 1
        completion = server.control("complete_negative_answer")
        assert completion["ok"] is True and completion["candidate_id"] == a_id
        a_completed = server.control("counters")["counters"]
        a_history = a_completed["attempt_history"][a_id]
        assert a_history["state"] == "final_queued"
        assert a_history["attention_active"] is False
        assert a_history["answer_kind"] == "no"
        assert a_completed["camera_meal_commits"] == 0
        delivered_history = a_delivered["attempt_history"][a_id]
        for field in (
            "request_identity", "request_ack", "image_sha256", "capture_time",
            "capture_time_authority", "photo_id", "photo_delivery_confirmed",
        ):
            assert a_history[field] == delivered_history[field]
        a_capture = a_post["request"]["capture_time"]
        query_id = candidate_id_for("id:restore-query", "rev-restore-query")
        status, a_projection_body = _references(
            server, token, query_id, a_capture, schema_version=2
        )
        assert status == 200
        a_projection = json.loads(a_projection_body)
        assert a_projection["schema_version"] == 2
        a_receipt = next(item for item in a_projection["references"]
                         if item["candidate_id"] == a_id)
        assert a_history["request_identity"] == a_post["request"]
        assert a_history["request_ack"]["status"] == 202
        assert a_history["request_ack"]["body"] == a_ack
        assert a_history["photo_delivery_confirmed"] is True
        a_artifacts = initial["publication_artifacts"]["id:tcp-older"]
        assert a_artifacts["candidate_id"] == a_id

        gap = server.control("make_restorable_legacy_source_gap")
        assert gap["ok"] is True and gap["candidate_id"] == a_id
        restarted = server.control("restart")
        assert restarted["journal_preserved"] is True
        server.port = restarted["port"]
        before_repair = server.control("counters")["counters"]
        assert before_repair["reference_source_records"] == 0
        assert before_repair["attempt_history"][a_id]["state"] == "final_queued"
        assert before_repair["attempt_history"][a_id]["attention_active"] is False
        assert before_repair["attempt_history"][a_id]["answer_kind"] == "no"
        assert before_repair["attempt_history"][a_id]["capture_time"] is None
        assert before_repair["attempt_history"][a_id]["capture_time_authority"] is None
        assert before_repair["attempt_history"][a_id]["request_identity"] == a_history[
            "request_identity"
        ]
        assert before_repair["attempt_history"][a_id]["request_ack"] == a_history[
            "request_ack"
        ]
        assert before_repair["attempt_history"][a_id]["photo_id"] == a_receipt[
            "native_photo_message_id"
        ]
        assert before_repair["attempt_history"][a_id]["photo_delivery_confirmed"] is True
        assert server.control("hold_photos")["ok"] is True

        distinct = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "distinct", "non_food"
        )
        assert distinct["scene_source_operations"] == 1, {
            "results": [
                (item["outcome"], item["submission_outcome"],
                 item["scene_source_operations"], item["scene_sources_reestablished"],
                 item["scene_reference_hold_count"])
                for item in distinct["results"]
            ],
            "http": [(item["path"], item["response_status"])
                     for item in distinct["observed"]],
            "source_operations": [
                (item["request"].get("candidate_id"), item["response_status"],
                 json.loads(item["response_body"]).get("error", {}).get("code"))
                for item in distinct["source_operations"]
            ],
            "queries": [
                (item["request"].get("candidate_id"), item["response_status"],
                 json.loads(item["response_body"]).get("coverage"),
                 json.loads(item["response_body"]).get("unresolved_candidate_ids"))
                for item in distinct["reference_queries"]
            ],
        }
        source_op = distinct["source_operations"][0]
        assert source_op["manifest_sha256"] == a_artifacts["manifest_sha256"]
        assert source_op["producer_sha256"] == a_artifacts["sidecar_sha256"]
        assert source_op["image_filename"] == a_artifacts["original_filename"]
        assert source_op["image_content_type"] == a_artifacts["mime_type"]
        expected_source_request = {
            "schema_version": 1,
            "candidate_id": a_id,
            "source_revision": a_post["request"]["source_revision"],
            "manifest_sha256": a_post["request"]["manifest_sha256"],
            "image_sha256": a_post["request"]["image_sha256"],
            "capture_time": a_post["request"]["capture_time"],
            "capture_time_authority": a_post["request"]["capture_time_authority"],
        }
        assert distinct["scene_sources_reestablished"] == 1, {
            "operation_status": source_op["response_status"],
            "source_request_matches_admission": (
                {key: value for key, value in source_op["request"].items()
                 if key != "capture_time"}
                == {key: value for key, value in expected_source_request.items()
                    if key != "capture_time"}
                and datetime.fromisoformat(
                    source_op["request"]["capture_time"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                == datetime.fromisoformat(
                    expected_source_request["capture_time"].replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            ),
            "source_request_field_differences": {
                key: (source_op["request"].get(key), expected_source_request.get(key))
                for key in expected_source_request
                if key != "capture_time"
                and source_op["request"].get(key) != expected_source_request.get(key)
            },
            "source_manifest_digest_matches_request": source_op["manifest_sha256"]
            == source_op["request"]["manifest_sha256"],
                "source_image_digest_matches_request": source_op["image_sha256"]
                == source_op["request"]["image_sha256"],
                "source_producer_digest_matches_publication": source_op["producer_sha256"]
                == a_artifacts["sidecar_sha256"],
                "source_manifest_digest_matches_publication": source_op["manifest_sha256"]
                == a_artifacts["manifest_sha256"],
                "source_request_schema": source_op["request"].get("schema_version"),
                "source_operation_body": source_op["response_body"],
        }
        assert distinct["scene_reference_hold_count"] == 0
        assert distinct["scene_provider_requests"] == 1
        assert len(distinct["source_operations"]) == 1
        assert source_op["response_status"] == 200
        assert json.loads(source_op["response_body"])["status"] == "source_reestablished"
        source_request = source_op["request"]
        assert {
            key: value for key, value in source_request.items() if key != "capture_time"
        } == {
            key: value for key, value in expected_source_request.items()
            if key != "capture_time"
        }
        assert datetime.fromisoformat(
            source_request["capture_time"].replace("Z", "+00:00")
        ).astimezone(timezone.utc) == datetime.fromisoformat(
            expected_source_request["capture_time"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        assert source_op["manifest_sha256"] == source_request["manifest_sha256"]
        assert source_op["image_sha256"] == source_request["image_sha256"]
        queries = distinct["reference_queries"]
        assert len(queries) == 2
        assert all(item["request"]["schema_version"] == 2 for item in queries)
        assert all(item["response_status"] == 200 for item in queries)
        incomplete = json.loads(queries[0]["response_body"])
        complete = json.loads(queries[1]["response_body"])
        assert incomplete["coverage"] == "incomplete"
        assert incomplete["unresolved_candidate_ids"] == [a_id]
        assert complete["coverage"] == "complete"
        assert complete["unresolved_candidate_ids"] == []
        restored_receipt = next(
            item for item in complete["reestablished_references"]
            if item["candidate_id"] == a_id
        )
        assert restored_receipt["native_photo_message_id"] == a_receipt[
            "native_photo_message_id"
        ]
        assert restored_receipt["image_sha256"] == a_receipt["image_sha256"]
        assert a_artifacts == distinct["publication_artifacts"]["id:tcp-older"]

        order = distinct["operation_order"]
        first_query = order.index("/internal/v1/camera/references")
        source_index = order.index("/internal/v1/camera/reference-source", first_query + 1)
        second_query = order.index("/internal/v1/camera/references", source_index + 1)
        b_admission = order.index("/internal/v1/camera/candidates", second_query + 1)
        session_lease = order.index("/internal/v1/camera/session")
        b_uploads = [item for item in distinct["observed"]
                     if item["path"] == "/internal/v1/camera/candidates"
                     and item["request"]["source_revision"] == "rev:tcp-b"]
        assert len(b_uploads) == 1 and b_uploads[0]["response_status"] == 202
        assert all(
            item["request"]["source_revision"] != "rev:tcp-a"
            for item in distinct["observed"]
            if item["path"] == "/internal/v1/camera/candidates"
        )
        assert b_uploads[0]["request"]["seq"] == 1
        after_restore = server.control("counters")["counters"]
        history_after = after_restore["attempt_history"][a_id]
        assert history_after["state"] == "final_queued"
        assert history_after["attention_active"] is False
        assert history_after["answer_kind"] == "no"
        assert history_after["request_identity"] == a_history["request_identity"]
        assert history_after["request_ack"] == a_history["request_ack"]
        assert history_after["photo_id"] == a_history["photo_id"]
        assert history_after["photo_delivery_confirmed"] is True
        assert after_restore["photo_calls"] == after_restore["confirmed_photo_attempts"] == 1
        assert after_restore["camera_meal_commits"] == 0
        assert after_restore["committed_seq"] == 1
        assert after_restore["journal_entries"] == 2
        assert b_admission > second_query
        assert session_lease > second_query

        assert server.control("release_photos")["ok"] is True
        after_b_delivery = server.control("await_idle")["counters"]
        assert after_b_delivery["photo_calls"] == after_b_delivery[
            "confirmed_photo_attempts"
        ] == 2
        assert after_b_delivery["journal_entries"] == 2
        assert after_b_delivery["camera_meal_commits"] == 0
    finally:
        server.close()


def test_actual_scene_producer_allows_distinct_and_suppresses_confirmed_repeat(tmp_path):
    checkout_value = os.environ.get("TELEGENT_CHECKOUT")
    python_value = os.environ.get("TELEGENT_PYTHON")
    if not checkout_value or not python_value:
        pytest.skip("set TELEGENT_CHECKOUT and TELEGENT_PYTHON for the cross-repo fixture")
    checkout = Path(checkout_value).resolve(strict=True)
    producer_python = Path(python_value).absolute()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    token_file = tmp_path / "camera.token"
    token_file.write_bytes(b"d" * 40 + b"\n")
    token_file.chmod(0o600)
    server = ServerProcess(workspace, token_file)
    token = "d" * 40
    producer_state = tmp_path / "producer-state"
    producer_state.mkdir(mode=0o700)
    _lease(server, token)
    producer_clock = server.control("counters")["counters"]["camera_now"]
    (producer_state / "capture-clock.json").write_text(
        json.dumps({"now": producer_clock}), encoding="utf-8"
    )
    try:
        initial = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "initial", "non_food"
        )
        assert any(item["path"] == "/internal/v1/camera/candidates"
                   for item in initial["observed"])
        first_delivery = server.control("await_idle")["counters"]
        assert first_delivery["photo_calls"] == first_delivery["confirmed_photo_attempts"] == 1
        assert server.control("complete_negative_answer")["ok"] is True

        distinct = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "distinct", "non_food"
        )
        b_uploads = [item for item in distinct["observed"]
                     if item["path"] == "/internal/v1/camera/candidates"
                     and item["request"]["source_revision"] == "rev:tcp-b"]
        assert len(b_uploads) == 1
        assert b_uploads[0]["response_status"] == 202, {
            "request": b_uploads[0]["request"],
            "response": b_uploads[0]["response_body"],
        }
        assert distinct["scene_provider_requests"] == 1
        after_b = server.control("await_idle")["counters"]
        assert after_b["photo_calls"] == after_b["confirmed_photo_attempts"] == 2
        assert after_b["journal_entries"] == 2
        assert server.control("complete_negative_answer")["ok"] is True

        repeat = _run_scene_stage(
            checkout, producer_python, producer_state, server, token, "repeat", "food"
        )
        c_id = repeat["target_candidate_id"]
        c_result = next(item for item in repeat["results"] if item["candidate_id"] == c_id)
        assert c_result["outcome"] == "scene_repeat_suppressed", {
            "candidate": c_result,
            "queries": repeat["reference_queries"],
            "observed": repeat["observed"],
            "source_operations": repeat["source_operations"],
            "provider_requests": repeat["scene_provider_requests"],
        }
        generated = producer_state / "generated"
        original_a, original_b, original_c = (
            (generated / name).read_bytes() for name in ("a.jpg", "b.jpg", "c.jpg")
        )
        fingerprint_a = fingerprint_image_bytes(original_a)
        fingerprint_b = fingerprint_image_bytes(original_b)
        fingerprint_c = fingerprint_image_bytes(original_c)
        assert len({
            fingerprint_a["sha256"], fingerprint_b["sha256"], fingerprint_c["sha256"]
        }) == 3
        assert fingerprint_a["sha256"] != fingerprint_c["sha256"]
        assert fingerprint_b["sha256"] != fingerprint_c["sha256"]
        for prior in (fingerprint_a, fingerprint_b):
            distance = phash_hamming_distance(prior["phash"], fingerprint_c["phash"])
            assert distance is not None and distance > PHASH_HAMMING_THRESHOLD
        assert repeat["scene_provider_requests"] == 2
        assert len(repeat["reference_queries"]) == 1
        assert repeat["reference_queries"][0]["request"]["schema_version"] == 2
        assert repeat["reference_queries"][0]["response_status"] == 200
        assert all(item["request"]["source_revision"] != "rev:tcp-c"
                   for item in repeat["observed"])
        after_c = server.control("counters")["counters"]
        assert after_c["photo_calls"] == after_c["confirmed_photo_attempts"] == 2
        assert after_c["journal_entries"] == 2
        assert after_c["committed_seq"] == after_b["committed_seq"]
        assert after_c["camera_meal_commits"] == 0
        assert c_id not in {
            item.get("candidate_id") for item in repeat["observed"]
        }
    finally:
        server.close()
