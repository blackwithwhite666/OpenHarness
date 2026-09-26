"""Offline Camera ingress contract; no Telegram, model, or private media calls."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.gateway.camera import (
    _PENDING_TTL_SECONDS,
    CAMERA_AUTHORITY,
    CameraCandidateUpload,
    CameraIngress,
    serve_camera_http,
)
from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.models import CameraIngressConfig, GatewayConfig
from ohmo.gateway.runtime import GatewayStreamUpdate, OhmoSessionRuntimePool
from ohmo.gateway.service import OhmoGatewayService
from ohmo.gateway.turn_context import TurnContext
from ohmo.gateway.memory_gate import MemoryScope
from ohmo.evals import get_eval_store
from ohmo.evals.recorder import GatewayEvalRecorder
from ohmo.camera_protocol.models import candidate_id_for
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.evals import DecisionTraceValidationError, TRACE_FINALIZATION


FIXTURE = Path(__file__).parents[2] / "ohmo/camera_protocol/manifest_v2_fixture.json"


def _deepseek_route_json() -> str:
    return json.dumps(
        {
            "candidate_identity_sha256": "1" * 64,
            "requested_model": "deepseek/deepseek-v4.1-flash",
            "requested_provider_only": ["deepinfra/fp8"],
            "requested_zdr": True,
            "requested_data_collection": "deny",
            "requested_allow_fallbacks": False,
            "requested_response_cache_header": "false",
            "prepared_provider_only": ["deepinfra/fp8"],
            "prepared_zdr": True,
            "prepared_data_collection": "deny",
            "prepared_allow_fallbacks": False,
            "prepared_provider_policy_source": "prepared_outbound_request_body",
            "prepared_response_cache_header": "false",
            "prepared_response_cache_header_source": "prepared_outbound_request_headers",
            "catalog_observed_at": "2026-09-25T00:00:00+00:00",
            "catalog_endpoint": "deepinfra/fp8",
            "catalog_zdr": True,
            "catalog_sha256": "2" * 64,
            "catalog_terms_sha256": "3" * 64,
            "requested_at": "2026-09-25T00:00:01+00:00",
            "response_received_at": "2026-09-25T00:00:02+00:00",
            "response_model": "deepseek/deepseek-v4.1-flash",
            "response_provider": "deepinfra",
            "response_endpoint": None,
            "response_endpoint_source": "unverified",
            "response_cache_status": "MISS",
            "response_cache_status_source": "response_header",
            "request_identity_sha256": "4" * 64,
            "response_id_sha256": "5" * 64,
            "response_identity_sha256": "6" * 64,
        },
        separators=(",", ":"),
    )


class FakeTelegram:
    polling_started = True

    def __init__(self, *, fail: bool = False, text_receipt: bool = False) -> None:
        self.fail = fail
        self.text_receipt = text_receipt
        self.calls: list[tuple[str, str]] = []

    async def send_camera_photo(self, *, chat_id: str, image_path: str, caption: str):
        self.calls.append((chat_id, image_path))
        if self.fail:
            raise RuntimeError("photo failed")
        return OutboundDeliveryReceipt(
            channel="telegram",
            chat_id=chat_id,
            native_message_ids=(("text-77",) if self.text_receipt else (77,)),
        )


def _candidate(root: Path, *, index: int = 0) -> dict:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["file_id"] = f"id:fake-{index}"
    payload["rev"] = f"rev-{index}"
    payload["candidate_id"] = candidate_id_for(payload["file_id"], payload["rev"])
    payload["event_id"] = f"{payload['candidate_id']}:manifest:v1"
    payload["classifier_model"] = "deepseek/deepseek-v4.1-flash"
    payload["classifier_policy_version"] = "deepseek-camera-production-v1"
    payload["classifier_route_attestation_json"] = _deepseek_route_json()
    image = b"fake-offline-image" + str(index).encode()
    payload["original_size_bytes"] = len(image)
    payload["original_sha256"] = hashlib.sha256(image).hexdigest()
    directory = root / payload["candidate_id"]
    directory.mkdir(parents=True)
    (directory / "original.jpg").write_bytes(image)
    manifest = json.dumps(payload, separators=(",", ":")).encode()
    (directory / "manifest.json").write_bytes(manifest)
    request = {
        "candidate_id": payload["candidate_id"],
        "source_revision": payload["rev"],
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "image_sha256": payload["original_sha256"],
        "capture_time": payload["normalized_capture_time"],
        "capture_time_authority": "exif",
    }
    _producer_sidecar(root, request)
    return request


def _ingress(tmp_path: Path, telegram: FakeTelegram | None = None):
    root = tmp_path / "artifacts"
    root.mkdir(exist_ok=True)
    token = tmp_path / "camera.token"
    token.write_text("s" * 40 + "\n", encoding="ascii")
    token.chmod(0o600)
    config = CameraIngressConfig(
        enabled=True,
        listen_port=8765,
        bearer_token_file=token,
        principal="123",
        tenant_id="marina",
        chat_id="123",
        session_key="telegram:123",
    )
    bus = MessageBus()
    channel = telegram or FakeTelegram()
    return CameraIngress(config, workspace=tmp_path, bus=bus, telegram=channel), root, bus, channel


def _upload(root: Path, request: dict) -> CameraCandidateUpload:
    candidate_dir = root / request["candidate_id"]
    manifest_bytes = (candidate_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    return CameraCandidateUpload(
        request=request,
        manifest_bytes=manifest_bytes,
        producer_sidecar_bytes=(
            root / "_producer" / f"{request['candidate_id']}.json"
        ).read_bytes(),
        image_bytes=(candidate_dir / "original.jpg").read_bytes(),
        image_filename=manifest["original_filename"],
        image_content_type=manifest["mime_type"],
    )


async def _leased_upload(
    ingress: CameraIngress, root: Path, authorization: str | None, request: dict
) -> CameraCandidateUpload:
    status, lease = await ingress.lease(authorization)
    if status != 200:
        raise AssertionError(f"test lease failed: {status} {lease}")
    payload = {
        **request,
        "session_id": lease["session_id"],
        "epoch": lease["epoch"],
        "seq": lease["committed_seq"] + 1,
    }
    return _upload(root, payload)


async def _admit(ingress: CameraIngress, root: Path, authorization: str | None, request: dict):
    status, lease = await ingress.lease(authorization)
    if status != 200:
        return status, lease
    payload = {
        **request,
        "session_id": lease["session_id"],
        "epoch": lease["epoch"],
        "seq": lease["committed_seq"] + 1,
    }
    return await ingress.admit(authorization, _upload(root, payload))


def _multipart(upload: CameraCandidateUpload) -> tuple[str, bytes]:
    boundary = "camera-test-boundary"
    fields = [
        ("request", None, "application/json", json.dumps(upload.request).encode()),
        ("manifest", None, "application/json", upload.manifest_bytes),
        ("producer", None, "application/json", upload.producer_sidecar_bytes),
        ("image", upload.image_filename, upload.image_content_type, upload.image_bytes),
    ]
    body = bytearray()
    for name, filename, content_type, value in fields:
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'.encode())
        if filename is not None:
            body.extend(f'; filename="{filename}"'.encode())
        body.extend(f"\r\nContent-Type: {content_type}\r\n\r\n".encode())
        body.extend(value)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def _producer_sidecar(root: Path, request: dict) -> Path:
    manifest = json.loads((root / request["candidate_id"] / "manifest.json").read_text())
    release = {
        "model": manifest["classifier_model"],
        "prompt_version": manifest["classifier_prompt_version"],
        "policy_version": manifest["classifier_policy_version"],
        "dataset_version": manifest["classifier_dataset_version"],
    }
    clip_release = {
        "model_id": "sentence-transformers/clip-ViT-B-32",
        "model_revision": "a" * 40,
        "preprocessing_version": "classifier-only-exif-orientation-resize-v1",
        "threshold": -0.028223291039466858,
    }
    sidecar = {
        "schema_version": 1,
        "candidate_id": request["candidate_id"],
        "revision": 1,
        "state": "published",
        "state_history": ["discovered", "classified", "published"],
        "release": release,
        "decision": {
            "candidate_id": request["candidate_id"],
            "model": release["model"],
            "prompt_version": release["prompt_version"],
            "policy_version": release["policy_version"],
            "classifier_route_attestation_json": manifest["classifier_route_attestation_json"],
            "output": manifest["classifier_output"],
            "is_food_like": True,
        },
        "clip_runtime_mode": "enforce",
        "clip_release": clip_release,
        "clip_decision": {
            **clip_release,
            "candidate_id": request["candidate_id"],
            "outcome": "pass",
            "forward_to_classifier": True,
            "score": clip_release["threshold"],
        },
    }
    directory = root / "_producer"
    directory.mkdir(exist_ok=True)
    path = directory / f"{request['candidate_id']}.json"
    path.write_text(json.dumps(sidecar), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_admission_photo_receipt_precedes_one_ordinary_synthetic_turn(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    status, body = await _admit(ingress, root, "Bearer " + "s" * 40, request)
    assert status == 202
    assert body == {
        "status": "admitted",
        "candidate_id": request["candidate_id"],
        "admission_id": body["admission_id"],
        "delivery_semantics": "in_process_only",
        "session_id": body["session_id"],
        "epoch": body["epoch"],
        "ack_seq": 1,
    }
    assert bus.inbound_size == 0  # 202 is admission, not a delivery claim.
    event = await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert channel.calls and event.sender_id == "__camera__"
    assert event.session_key == "telegram:123"
    assert event.metadata["_camera_photo_id"] == 77
    assert event.metadata["_camera_authority"] is CAMERA_AUTHORITY
    assert len(event.media) == 1 and Path(event.media[0]).read_bytes() == b"fake-offline-image0"
    assert bus.inbound_size == 0
    assert ingress._attempts[request["candidate_id"]]["state"] == "photo_sent"
    await ingress.close()


@pytest.mark.asyncio
async def test_auth_allowlist_and_evidence_fail_closed_before_send(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    for bad in (None, "Bearer wrong", "Basic " + "s" * 40):
        assert (await ingress.admit(bad, request))[0] == 401
    for field in ("chat_id", "session_key", "sender_id", "path", "prompt", "trusted", "consumed"):
        assert (await _admit(ingress, root, "Bearer " + "s" * 40, {**request, field: "forged"}))[
            0
        ] == 400
    assert (
        await _admit(ingress, root, "Bearer " + "s" * 40, {**request, "image_sha256": "0" * 64})
    )[0] == 422
    assert (
        await _admit(ingress, root, "Bearer " + "s" * 40, {**request, "manifest_sha256": "0" * 64})
    )[0] == 422
    assert (
        await _admit(
            ingress,
            root,
            "Bearer " + "s" * 40,
            {**request, "capture_time": "2026-08-05T01:00:00Z"},
        )
    )[0] == 422
    assert channel.calls == [] and bus.inbound_size == 0 and not ingress._attempts


@pytest.mark.asyncio
async def test_uploaded_image_filename_cannot_escape_candidate_name(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    upload = replace(upload, image_filename="../outside.jpg")
    assert (await ingress.admit("Bearer " + "s" * 40, upload))[0] == 422
    assert channel.calls == [] and bus.inbound_size == 0


@pytest.mark.asyncio
async def test_candidate_manifest_mime_mismatch_is_rejected(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    path = root / request["candidate_id"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["mime_type"] = "application/pdf"
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    path.write_bytes(raw)
    request["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 422
    assert channel.calls == [] and bus.inbound_size == 0


@pytest.mark.asyncio
async def test_direct_upload_does_not_depend_on_dropbox_artifact_sync(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    (root / request["candidate_id"] / "manifest.json").unlink()
    (root / request["candidate_id"] / "original.jpg").unlink()
    (root / "_producer" / f"{request['candidate_id']}.json").unlink()
    assert (await ingress.admit("Bearer " + "s" * 40, upload))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert len(channel.calls) == 1
    await ingress.close()


@pytest.mark.asyncio
async def test_json_candidate_payload_is_not_an_admission_contract(
    tmp_path: Path,
) -> None:
    ingress, root, _, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)

    status, body = await ingress.admit("Bearer " + "s" * 40, upload.request)

    assert (status, body) == (400, {"error": {"code": "invalid_request"}})
    assert ingress._session["committed_seq"] == 0
    assert channel.calls == []
    await ingress.close()


@pytest.mark.asyncio
async def test_same_sequence_replays_cached_admission_without_second_send(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    first = await ingress.admit("Bearer " + "s" * 40, upload)
    duplicate = await ingress.admit("Bearer " + "s" * 40, upload)
    assert first == duplicate
    assert first[0] == 202 and first[1]["ack_seq"] == 1
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert len(channel.calls) == 1
    await ingress.close()


@pytest.mark.asyncio
async def test_sequence_gap_returns_expected_seq_without_consuming_it(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    out_of_order = replace(upload, request={**upload.request, "seq": 2})
    status, body = await ingress.admit("Bearer " + "s" * 40, out_of_order)
    assert status == 409
    assert body == {"error": {"code": "expected_seq"}, "expected_seq": 1}
    assert ingress._session["committed_seq"] == 0
    assert (await ingress.admit("Bearer " + "s" * 40, upload))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    await ingress.close()


@pytest.mark.asyncio
async def test_final_evidence_rejection_commits_sequence_and_is_cached(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    invalid = replace(upload, image_bytes=b"tampered")
    first = await ingress.admit("Bearer " + "s" * 40, invalid)
    duplicate = await ingress.admit("Bearer " + "s" * 40, invalid)
    assert first == duplicate
    assert first[0] == 422 and first[1]["ack_seq"] == 1
    next_request = _candidate(root, index=1)
    accepted = await _admit(ingress, root, "Bearer " + "s" * 40, next_request)
    assert accepted[0] == 202 and accepted[1]["ack_seq"] == 2
    assert len(channel.calls) == 0
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    await ingress.close()


@pytest.mark.asyncio
async def test_expired_lease_rotates_epoch_and_rejects_old_request(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    old_upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    old_epoch = old_upload.request["epoch"]
    ingress._session["expires_at"] = "2000-01-01T00:00:00+00:00"
    status, lease = await ingress.lease("Bearer " + "s" * 40)
    assert status == 200
    assert lease["epoch"] != old_epoch and lease["committed_seq"] == 0
    assert (await ingress.admit("Bearer " + "s" * 40, old_upload))[1]["error"][
        "code"
    ] == "session_expired"
    fresh_upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    assert (await ingress.admit("Bearer " + "s" * 40, fresh_upload))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    await ingress.close()


def test_camera_listener_accepts_only_loopback_or_selected_vpn_address() -> None:
    assert CameraIngressConfig().enabled is False
    assert CameraIngressConfig().listen_host == "127.0.0.1"
    selected = CameraIngressConfig(listen_host="10.8.0.8", listen_port=18751)
    assert (selected.listen_host, selected.listen_port) == ("10.8.0.8", 18751)
    for host in ("0.0.0.0", "10.8.0.7", "8.8.8.8", "::", "example.com"):
        with pytest.raises(ValueError):
            CameraIngressConfig(listen_host=host)


@pytest.mark.asyncio
async def test_unrecognized_or_unbound_real_text_is_nutrition_forbidden(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    for text, target in (
        # Explicit answers (anchored or bare consumption language) bind and
        # are covered by the binding tests; these must stay nutrition-forbidden.
        ("посмотри ещё раз", 77),
        ("не знаю", None),
        ("только сливы", None),  # scope-only binds only when anchored
        ("Как погода?", None),
        ("Обычный разговор", 999),
    ):
        message = InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content=text,
            metadata={"is_group": False, "reply_to_message_id": target, "_telegram_raw_text": text},
        )
        ingress.process_real_inbound(message)
        assert message.metadata.get("_camera_unbound") is CAMERA_AUTHORITY
        assert message.metadata.get("_camera_answer") is None
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["missing", "negative", "wrong_release", "wrong_clip", "wrong_route"]
)
async def test_producer_publication_evidence_required_before_send(
    tmp_path: Path, mutation: str
) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    path = _producer_sidecar(root, request)
    if mutation == "missing":
        upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
        path.unlink()
        upload = replace(upload, producer_sidecar_bytes=b"")
        result = await ingress.admit("Bearer " + "s" * 40, upload)
    else:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "negative":
            sidecar["state"] = "negative"
        elif mutation == "wrong_release":
            sidecar["release"]["model"] = "other-model"
        elif mutation == "wrong_route":
            manifest_path = root / request["candidate_id"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            route = json.loads(manifest["classifier_route_attestation_json"])
            route["requested_zdr"] = False
            manifest["classifier_route_attestation_json"] = json.dumps(route)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            request["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        else:
            sidecar["clip_decision"]["model_revision"] = "b" * 40
        path.write_text(json.dumps(sidecar), encoding="utf-8")
        result = await _admit(ingress, root, "Bearer " + "s" * 40, request)
    assert result[0] == 422
    assert channel.calls == [] and bus.inbound_size == 0


@pytest.mark.asyncio
async def test_noncanonical_producer_manifest_event_is_not_published_evidence(
    tmp_path: Path,
) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    path = root / request["candidate_id"] / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["event_id"] = "forged-manifest-event"
    raw = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(raw)
    request["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 422
    assert channel.calls == [] and bus.inbound_size == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["state", "snapshots"])
async def test_symlinked_write_directory_fails_before_admission(
    tmp_path: Path, location: str
) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    state = tmp_path / "camera_ingress"
    if location == "state":
        state.symlink_to(outside, target_is_directory=True)
    else:
        state.mkdir()
        (state / "snapshots").symlink_to(outside, target_is_directory=True)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 503
    assert list(outside.iterdir()) == []
    assert channel.calls == [] and bus.inbound_size == 0


@pytest.mark.asyncio
async def test_duplicate_second_candidate_and_restart_never_resend(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    first = _candidate(root)
    second = _candidate(root, index=1)
    status, admitted = await _admit(ingress, root, "Bearer " + "s" * 40, first)
    assert status == 202
    duplicate_status, duplicate = await _admit(ingress, root, "Bearer " + "s" * 40, first)
    assert duplicate_status == 202
    assert duplicate["admission_id"] == admitted["admission_id"]
    assert duplicate["ack_seq"] == 2
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[1]["error"][
        "code"
    ] == "unresolved_candidate"
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    await ingress.close()
    reopened = CameraIngress(ingress.config, workspace=tmp_path, bus=MessageBus(), telegram=channel)
    reopened.mark_restart_unknown()
    assert (await _admit(reopened, root, "Bearer " + "s" * 40, first))[0] == 202
    assert reopened._attempts[first["candidate_id"]]["state"] == "delivery_unknown"
    assert len(channel.calls) == 1
    stale = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я это съела",
        metadata={
            "is_group": False,
            "reply_to_message_id": 77,
            "_telegram_raw_text": "Я это съела",
        },
    )
    reopened.process_real_inbound(stale)
    assert stale.metadata["_camera_unbound"] is CAMERA_AUTHORITY


@pytest.mark.asyncio
async def test_crash_after_202_leaves_tombstone_without_auto_retry(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    snapshot = Path(ingress._attempts[request["candidate_id"]]["snapshot"])
    assert snapshot.exists()
    await ingress.close()  # cancel before the scheduled worker executes
    assert bus.inbound_size == 0
    restarted = CameraIngress(ingress.config, workspace=tmp_path, bus=bus, telegram=channel)
    restarted.mark_restart_unknown()
    assert snapshot.exists()  # unresolved attempts retain their image for diagnosis/recovery
    status, duplicate = await _admit(restarted, root, "Bearer " + "s" * 40, request)
    assert status == 202 and duplicate["ack_seq"] == 1
    assert duplicate["admission_id"] == ingress._attempts[request["candidate_id"]]["admission_id"]
    assert channel.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fail,text_receipt", [(True, False), (False, True)])
async def test_failed_photo_or_fallback_text_never_dispatches_or_retries(
    tmp_path: Path, fail: bool, text_receipt: bool
) -> None:
    ingress, root, bus, channel = _ingress(
        tmp_path, FakeTelegram(fail=fail, text_receipt=text_receipt)
    )
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await asyncio.sleep(0)
    assert bus.inbound_size == 0
    assert len(channel.calls) == 1
    assert ingress._attempts[request["candidate_id"]]["state"] == "delivery_unknown"
    await ingress.close()


@pytest.mark.asyncio
async def test_explicit_real_reply_target_not_bare_yes_stale_or_other_user(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    snapshot = Path(ingress._attempts[request["candidate_id"]]["snapshot"])
    assert snapshot.exists()
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    def incoming(text: str, *, target: int | None = None, sender: str = "123") -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id=sender,
            chat_id="123",
            content=text,
            metadata={"is_group": False, "reply_to_message_id": target, "_telegram_raw_text": text},
        )

    for message in (
        incoming("да"),
        # A bare explicit consumption statement binds now (operator-approved
        # camera answer binding); see test_bare_explicit_answer_binds below.
        incoming("посмотри ещё раз"),
        incoming("да, я это съела", target=999),
        incoming("да, я это съела", target=77, sender="456"),
    ):
        ingress.process_real_inbound(message)
        assert message.metadata.get("_camera_answer") is None
    callback = incoming("Да, я это съела", target=77)
    callback.metadata["callback_query"] = True
    ingress.process_real_inbound(callback)
    assert callback.metadata.get("_camera_answer") is None
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="reply",
            metadata={
                "_camera_candidate_id": request["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(88,)),
    )
    assistant_reply = incoming("Я это съела", target=88)
    ingress.process_real_inbound(assistant_reply)
    assert assistant_reply.metadata["_camera_answer"] == "yes"
    assert ingress._attempts[request["candidate_id"]]["state"] == "answering"
    bound = incoming("Да, я это съела", target=77)
    ingress.process_real_inbound(bound)
    assert bound.metadata.get("_camera_answer") is None  # second answer cannot double-record
    assert assistant_reply.metadata["_camera_authority"] is CAMERA_AUTHORITY
    assert len(assistant_reply.media) == 1
    ingress.complete(assistant_reply, recorded=False)
    assert ingress._attempts[request["candidate_id"]]["state"] == "answering"
    ingress.complete(assistant_reply, recorded=True)
    assert ingress._attempts[request["candidate_id"]]["state"] == "final_queued"
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Записано",
            metadata={
                "_camera_candidate_id": request["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
                "_camera_final": CAMERA_AUTHORITY,
                "_camera_turn_id": assistant_reply.metadata["_camera_turn_id"],
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(89,)),
    )
    assert ingress._attempts[request["candidate_id"]]["state"] == "completed"
    assert not snapshot.exists()
    # A crash after persisting the terminal state but before unlinking is
    # repaired when the ingress starts again.
    snapshot.write_bytes(b"stale completed image")
    CameraIngress(ingress.config, workspace=tmp_path, bus=MessageBus(), telegram=None)
    assert not snapshot.exists()
    stale_reply = incoming("Я это съела", target=77)
    ingress.process_real_inbound(stale_reply)
    assert stale_reply.metadata["_camera_unbound"] is CAMERA_AUTHORITY
    stale_callback = incoming("Да, я это съела")
    stale_callback.metadata.update(callback_query=True, native_message_id=88)
    ingress.process_real_inbound(stale_callback)
    assert stale_callback.metadata["_camera_unbound"] is CAMERA_AUTHORITY
    stale_bare = incoming("да")
    ingress.process_real_inbound(stale_bare)
    assert stale_bare.metadata["_camera_unbound"] is CAMERA_AUTHORITY
    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    await ingress.close()


@pytest.mark.asyncio
async def test_bare_explicit_answer_binds_and_ambiguous_stays_unbound(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    snapshot = Path(ingress._attempts[request["candidate_id"]]["snapshot"])
    assert snapshot.exists()
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    def incoming(text: str) -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content=text,
            metadata={"is_group": False, "_telegram_raw_text": text},
        )

    ambiguous = incoming("посмотри ещё раз")
    ingress.process_real_inbound(ambiguous)
    assert ambiguous.metadata.get("_camera_answer") is None
    assert ambiguous.metadata["_camera_unbound"] is CAMERA_AUTHORITY

    bare_affirmation = incoming("да")
    ingress.process_real_inbound(bare_affirmation)
    assert bare_affirmation.metadata.get("_camera_answer") is None
    assert bare_affirmation.metadata["_camera_unbound"] is CAMERA_AUTHORITY

    bare_scope = incoming("Только сливы")
    ingress.process_real_inbound(bare_scope)
    assert bare_scope.metadata.get("_camera_answer") is None  # scope binds only anchored
    assert bare_scope.metadata["_camera_unbound"] is CAMERA_AUTHORITY

    bare_consumption = incoming("Я съела 4")
    ingress.process_real_inbound(bare_consumption)
    assert bare_consumption.metadata["_camera_answer"] == "yes"
    assert bare_consumption.metadata["_camera_authority"] is CAMERA_AUTHORITY
    assert bare_consumption.metadata["_camera_turn_id"]
    assert len(bare_consumption.media) == 1
    assert ingress._attempts[request["candidate_id"]]["state"] == "answering"

    ingress.complete(bare_consumption, recorded=True)
    assert ingress._attempts[request["candidate_id"]]["state"] == "final_queued"
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Записано",
            metadata={
                "_camera_candidate_id": request["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
                "_camera_final": CAMERA_AUTHORITY,
                "_camera_turn_id": bare_consumption.metadata["_camera_turn_id"],
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(89,)),
    )
    assert ingress._attempts[request["candidate_id"]]["state"] == "completed"

    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    bare_negation = incoming("Я это не ела")
    ingress.process_real_inbound(bare_negation)
    assert bare_negation.metadata["_camera_answer"] == "no"
    assert len(bare_negation.media) == 0
    assert ingress._attempts[second["candidate_id"]]["state"] == "answering"
    await ingress.close()


@pytest.mark.asyncio
async def test_ask_callback_answer_binds_and_foreign_callback_rejected(
    tmp_path: Path,
) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    snapshot = Path(ingress._attempts[request["candidate_id"]]["snapshot"])
    assert snapshot.exists()
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Вы это ели?",
            metadata={
                "_camera_candidate_id": request["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(88,)),
    )

    def callback(label: str, *, target: int, data: str = "ask:1") -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content=label,
            metadata={
                "is_group": False,
                "callback_query": True,
                "native_message_id": target,
                "callback_data": data,
            },
        )

    foreign_target = callback("Да, всё на фото", target=999)
    ingress.process_real_inbound(foreign_target)
    assert foreign_target.metadata["_camera_unbound"] is CAMERA_AUTHORITY

    foreign_data = callback("Да", target=77, data="menu:2")
    ingress.process_real_inbound(foreign_data)
    assert foreign_data.metadata["_camera_unbound"] is CAMERA_AUTHORITY

    affirmation = callback("Да, всё на фото", target=88)
    ingress.process_real_inbound(affirmation)
    assert affirmation.metadata["_camera_answer"] == "yes"
    assert affirmation.metadata["_camera_authority"] is CAMERA_AUTHORITY
    assert len(affirmation.media) == 1
    assert ingress._attempts[request["candidate_id"]]["state"] == "answering"

    ingress.complete(affirmation, recorded=True)
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Записано",
            metadata={
                "_camera_candidate_id": request["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
                "_camera_final": CAMERA_AUTHORITY,
                "_camera_turn_id": affirmation.metadata["_camera_turn_id"],
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(89,)),
    )
    assert ingress._attempts[request["candidate_id"]]["state"] == "completed"

    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    ingress.note_assistant_receipt(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Вы это ели?",
            metadata={
                "_camera_candidate_id": second["candidate_id"],
                "_camera_authority": CAMERA_AUTHORITY,
            },
        ),
        OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(90,)),
    )
    scope = callback("Только сливы", target=90, data="ask:1")
    ingress.process_real_inbound(scope)
    assert scope.metadata["_camera_answer"] == "yes"  # anchored scope-only binds
    assert len(scope.media) == 1
    await ingress.close()


@pytest.mark.asyncio
async def test_pending_ttl_sweeps_stale_answer_gate(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    snapshot = Path(ingress._attempts[request["candidate_id"]]["snapshot"])
    assert snapshot.exists()
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    candidate_id = request["candidate_id"]
    assert ingress._attempts[candidate_id]["state"] == "photo_sent"
    assert isinstance(ingress._attempts[candidate_id].get("admitted_at"), str)

    aged = datetime.now(UTC) - timedelta(seconds=_PENDING_TTL_SECONDS + 60)
    ingress._attempts[candidate_id]["admitted_at"] = aged.isoformat()
    gate = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="что нового",
        metadata={"is_group": False, "_telegram_raw_text": "что нового"},
    )
    ingress.process_real_inbound(gate)
    assert candidate_id not in ingress._attempts
    assert not snapshot.exists()
    assert gate.metadata.get("_camera_unbound") is None
    assert gate.metadata.get("_camera_authority") is None

    fresh = CameraIngress(ingress.config, workspace=tmp_path, bus=MessageBus(), telegram=None)
    assert candidate_id not in fresh._attempts

    # At-most-once tombstones survive the TTL.
    tomb = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, tomb))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    ingress._attempts[tomb["candidate_id"]]["state"] = "delivery_unknown"
    ingress._attempts[tomb["candidate_id"]]["admitted_at"] = aged.isoformat()
    ingress._sweep_expired_attempts()
    assert tomb["candidate_id"] in ingress._attempts

    # A legacy journal entry without admitted_at is backfilled, not swept.
    ingress._attempts[tomb["candidate_id"]].pop("admitted_at")
    ingress._save_attempts()
    reloaded = CameraIngress(ingress.config, workspace=tmp_path, bus=MessageBus(), telegram=None)
    assert tomb["candidate_id"] in reloaded._attempts
    assert isinstance(reloaded._attempts[tomb["candidate_id"]].get("admitted_at"), str)
    await ingress.close()


@pytest.mark.asyncio
async def test_second_candidate_waits_for_first_final_native_receipt(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    first = _candidate(root)
    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, first))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    answer = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я это съела",
        metadata={"reply_to_message_id": 77, "_telegram_raw_text": "Я это съела"},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_answer"] == "yes"
    ingress.complete(answer, recorded=True)  # runtime has not queued/sent final yet
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[1]["error"][
        "code"
    ] == "unresolved_candidate"
    progress = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Обрабатываю",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": first["candidate_id"],
        },
    )
    final = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Записано",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": first["candidate_id"],
            "_camera_final": CAMERA_AUTHORITY,
            "_camera_turn_id": answer.metadata["_camera_turn_id"],
        },
    )

    service = SimpleNamespace(_camera_ingress=ingress)
    receipt = OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(101,))
    await OhmoGatewayService._on_outbound_send_success(service, progress, receipt)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 409
    duplicate = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я это съела",
        metadata={"reply_to_message_id": 77, "_telegram_raw_text": "Я это съела"},
    )
    ingress.process_real_inbound(duplicate)
    assert duplicate.metadata["_camera_unbound"] is CAMERA_AUTHORITY
    assert duplicate.metadata.get("_camera_answer") is None
    await OhmoGatewayService._on_outbound_send_success(service, final, receipt)
    assert ingress._attempts[first["candidate_id"]]["state"] == "completed"
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 202
    await ingress.close()


@pytest.mark.asyncio
async def test_stale_camera_prompt_receipt_cannot_release_answer_turn(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    first = _candidate(root)
    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, first))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)

    # The model's initial Camera prompt is a final send too, but it belongs to
    # the analysis turn, not the later Marina answer/finalization turn.
    prompt = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Ты это съела?",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": first["candidate_id"],
            "_camera_final": CAMERA_AUTHORITY,
        },
    )
    answer = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я это съела",
        metadata={"reply_to_message_id": 77, "_telegram_raw_text": "Я это съела"},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_answer"] == "yes"
    ingress.complete(answer, recorded=True)

    service = SimpleNamespace(_camera_ingress=ingress)
    receipt = OutboundDeliveryReceipt(channel="telegram", chat_id="123", native_message_ids=(101,))
    wrong_candidate = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Записано",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": second["candidate_id"],
            "_camera_final": CAMERA_AUTHORITY,
            "_camera_turn_id": answer.metadata["_camera_turn_id"],
        },
    )
    await OhmoGatewayService._on_outbound_send_success(service, wrong_candidate, receipt)
    assert ingress._attempts[first["candidate_id"]]["state"] == "final_queued"
    await OhmoGatewayService._on_outbound_send_success(service, prompt, receipt)
    assert ingress._attempts[first["candidate_id"]]["state"] == "final_queued"

    final = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Записано",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": first["candidate_id"],
            "_camera_final": CAMERA_AUTHORITY,
            "_camera_turn_id": answer.metadata["_camera_turn_id"],
        },
    )
    await OhmoGatewayService._on_outbound_send_success(service, final, receipt)
    assert ingress._attempts[first["candidate_id"]]["state"] == "completed"
    await ingress.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failed", "missing_receipt", "text_receipt", "crash"])
async def test_final_send_failure_or_unknown_remains_unresolved(
    tmp_path: Path, outcome: str
) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    first = _candidate(root)
    second = _candidate(root, index=1)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, first))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    answer = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Нет, не ела",
        metadata={"reply_to_message_id": 77, "_telegram_raw_text": "Нет, не ела"},
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_answer"] == "no"
    ingress.complete(answer, recorded=False)
    final = OutboundMessage(
        channel="telegram",
        chat_id="123",
        content="Поняла, не записываю",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": first["candidate_id"],
            "_camera_final": CAMERA_AUTHORITY,
            "_camera_turn_id": answer.metadata["_camera_turn_id"],
        },
    )

    service = SimpleNamespace(_camera_ingress=ingress)
    if outcome == "crash":
        await ingress.close()
        ingress = CameraIngress(
            ingress.config, workspace=tmp_path, bus=MessageBus(), telegram=channel
        )
        ingress.mark_restart_unknown()
    elif outcome == "failed":
        await OhmoGatewayService._on_outbound_send_failure(service, final, RuntimeError("offline"))
    else:
        receipt = (
            None
            if outcome == "missing_receipt"
            else OutboundDeliveryReceipt(
                channel="telegram", chat_id="123", native_message_ids=("text-id",)
            )
        )
        await OhmoGatewayService._on_outbound_send_success(service, final, receipt)
    assert ingress._attempts[first["candidate_id"]]["state"] != "completed"
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, second))[0] == 409
    assert len(channel.calls) == 1
    await ingress.close()


@pytest.mark.asyncio
async def test_bridge_marks_only_camera_final_for_completion() -> None:
    candidate_id = "dropbox-camera-v1-" + "a" * 64

    class Runtime:
        async def stream_message(self, message, session_key):
            yield GatewayStreamUpdate(
                kind="assistant_update",
                text="Анализирую",
                metadata={
                    "_camera_authority": CAMERA_AUTHORITY,
                    "_camera_candidate_id": candidate_id,
                },
            )
            yield GatewayStreamUpdate(kind="final", text="Готово", metadata={})

    bus = MessageBus()
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=Runtime())
    camera = InboundMessage(
        channel="telegram",
        sender_id="__camera__",
        chat_id="123",
        content="photo",
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": candidate_id,
            "_camera_turn_id": "camera-turn",
        },
    )
    await bridge._process_message(camera, "telegram:123")
    progress = await bus.consume_outbound()
    final = await bus.consume_outbound()
    assert progress.metadata.get("_camera_final") is None
    assert final.metadata["_camera_final"] is CAMERA_AUTHORITY
    assert final.metadata["_camera_candidate_id"] == candidate_id
    assert final.metadata["_camera_turn_id"] == "camera-turn"

    manual = InboundMessage(channel="telegram", sender_id="123", chat_id="123", content="hi")
    await bridge._process_message(manual, "telegram:123")
    await bus.consume_outbound()
    manual_final = await bus.consume_outbound()
    assert manual_final.metadata.get("_camera_final") is None


@pytest.mark.asyncio
async def test_camera_debug_progress_cannot_replace_required_final_receipt() -> None:
    candidate_id = "dropbox-camera-v1-" + "a" * 64

    class Runtime:
        async def stream_message(self, message, session_key):
            yield GatewayStreamUpdate(kind="assistant_update", text="Готово", metadata={})
            yield GatewayStreamUpdate(
                kind="final", text="Готово", metadata={"_camera_turn_id": "forged-turn"}
            )

    bus = MessageBus()
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=Runtime(), debug_progress_chats=["123"])
    camera = InboundMessage(
        channel="telegram",
        sender_id="__camera__",
        chat_id="123",
        content="photo",
        metadata={"_camera_authority": CAMERA_AUTHORITY, "_camera_candidate_id": candidate_id},
    )
    await bridge._process_message(camera, "telegram:123")
    assert bus.outbound_size == 2
    progress = await bus.consume_outbound()
    final = await bus.consume_outbound()
    assert progress.metadata["_collapse"] is True
    assert progress.metadata.get("_camera_final") is None
    assert final.metadata["_camera_final"] is CAMERA_AUTHORITY
    assert final.metadata.get("_camera_turn_id") is None


@pytest.mark.asyncio
async def test_native_photo_reply_binds_and_manual_photo_is_untouched(tmp_path: Path) -> None:
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    manual = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я съела это",
        media=[str(tmp_path / "independent-manual.jpg")],
        metadata={"is_group": False, "_telegram_raw_text": "Я съела это"},
    )
    ingress.process_real_inbound(manual)
    assert manual.metadata.get("_camera_authority") is None
    assert len(manual.media) == 1
    replied_media = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я съела это",
        media=[str(tmp_path / "another.jpg")],
        metadata={
            "is_group": False,
            "reply_to_message_id": 77,
            "_telegram_raw_text": "Я съела это",
        },
    )
    ingress.process_real_inbound(replied_media)
    assert replied_media.metadata["_camera_unbound"] is CAMERA_AUTHORITY
    answer = InboundMessage(
        channel="telegram",
        sender_id="123|marina",
        chat_id="123",
        content="[quoted] Я это съела",
        metadata={
            "is_group": False,
            "reply_to_message_id": 77,
            "_telegram_raw_text": "Я это съела",
        },
    )
    ingress.process_real_inbound(answer)
    assert answer.metadata["_camera_answer"] == "yes"
    assert answer.metadata["_camera_candidate_id"] == request["candidate_id"]
    assert len(answer.media) == 1
    await ingress.close()


def test_disabled_config_and_generic_tenant_binding(tmp_path: Path) -> None:
    assert GatewayConfig().camera_ingress.enabled is False
    disabled = CameraIngress(
        CameraIngressConfig(enabled=False),
        workspace=tmp_path,
        bus=MessageBus(),
        telegram=FakeTelegram(),
    )
    assert asyncio.run(disabled.admit(None, {})) == (403, {"error": {"code": "camera_disabled"}})
    config = CameraIngressConfig(
        enabled=True,
        listen_port=8765,
        bearer_token_file=tmp_path / "token",
        principal="123",
        tenant_id="family",
        chat_id="123",
        session_key="telegram:123",
    )
    validated = GatewayConfig(
        enabled_channels=["telegram"],
        conversation_learning=True,
        evals_capture=True,
        memory_backend="shadow",
        family_principals={"123": "family"},
        enabled_memory_tenants=("family",),
        camera_ingress=config,
        honcho_base_url="https://honcho.test",
        tenant_honcho={"family": {"workspace": "w", "api_key": "a", "observed_peer": "p"}},
    )
    assert validated.camera_ingress.tenant_id == "family"


def test_camera_synthetic_does_not_bind_session_owner(tmp_path: Path) -> None:
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = SimpleNamespace(
        camera_ingress=SimpleNamespace(tenant_id="family"),
        owner_principals=(),
        family_principals={"123": "family"},
        enabled_memory_tenants=("family",),
        shared_tenants=(),
    )
    pool._session_owner_principals = {}
    turn = TurnContext(
        principal="__camera__",
        is_owner=False,
        is_private=True,
        channel="telegram",
        chat_id="123",
        session_id="session",
        camera_authorized=True,
    )
    synthetic = InboundMessage(
        channel="telegram",
        sender_id="__camera__",
        chat_id="123",
        content="photo",
        metadata={"_synthetic": True, "is_group": False},
    )
    assert pool._bind_session_owner(synthetic, "telegram:123", turn) is None
    assert pool._session_owner_principals == {}
    real = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="hi",
        metadata={"is_group": False},
    )
    real_turn = TurnContext(
        principal="123",
        is_owner=False,
        is_private=True,
        channel="telegram",
        chat_id="123",
        session_id="session",
    )
    assert pool._bind_session_owner(real, "telegram:123", real_turn) == "123"
    assert pool._honcho_turn_allowed(turn, MemoryScope(private_tenant="family", shared_tenants=()))
    assert not pool._honcho_turn_allowed(
        turn, MemoryScope(private_tenant="other", shared_tenants=())
    )


def test_classifier_only_and_unbound_turn_cannot_finalize_meal(tmp_path: Path) -> None:
    recorder = GatewayEvalRecorder(store=get_eval_store(tmp_path), episode_id="offline-camera")
    recorder.forbid_nutrition_record()
    with pytest.raises(DecisionTraceValidationError, match="bound explicit Marina answer"):
        recorder.decision_trace_recorder.record(
            TRACE_FINALIZATION,
            {"annotations": {"nutrition": {"record_type": "meal_observation"}}},
        )
    assert recorder.validated_nutrition_envelope is None


class _Writer:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


class _LostResponseWriter(_Writer):
    async def drain(self) -> None:
        raise ConnectionError("response lost after admission")


@pytest.mark.asyncio
async def test_http_exact_route_and_error_shape(tmp_path: Path) -> None:
    ingress, root, _, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    content_type, body = _multipart(upload)
    reader = asyncio.StreamReader()
    reader.feed_data(
        b"POST /internal/v1/camera/candidates HTTP/1.1\r\n"
        + b"Authorization: Bearer "
        + b"s" * 40
        + b"\r\n"
        + f"Content-Type: {content_type}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    reader.feed_eof()
    writer = _Writer()
    await serve_camera_http(ingress, reader, writer)
    assert writer.data.startswith(b"HTTP/1.1 202 ")
    response = json.loads(writer.data.split(b"\r\n\r\n", 1)[1])
    assert response["delivery_semantics"] == "in_process_only"
    assert response["ack_seq"] == 1
    await ingress.close()
    assert len(channel.calls) <= 1


@pytest.mark.asyncio
async def test_lost_http_response_after_admission_never_causes_resend(tmp_path: Path) -> None:
    ingress, root, bus, channel = _ingress(tmp_path)
    request = _candidate(root)
    upload = await _leased_upload(ingress, root, "Bearer " + "s" * 40, request)
    content_type, body = _multipart(upload)
    reader = asyncio.StreamReader()
    reader.feed_data(
        b"POST /internal/v1/camera/candidates HTTP/1.1\r\n"
        + b"Authorization: Bearer "
        + b"s" * 40
        + b"\r\n"
        + f"Content-Type: {content_type}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    reader.feed_eof()
    with pytest.raises(ConnectionError, match="response lost"):
        await serve_camera_http(ingress, reader, _LostResponseWriter())

    retry_reader = asyncio.StreamReader()
    retry_reader.feed_data(
        b"POST /internal/v1/camera/candidates HTTP/1.1\r\n"
        + b"Authorization: Bearer "
        + b"s" * 40
        + b"\r\n"
        + f"Content-Type: {content_type}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    retry_reader.feed_eof()
    retry_writer = _Writer()
    await serve_camera_http(ingress, retry_reader, retry_writer)
    assert retry_writer.data.startswith(b"HTTP/1.1 202 ")
    replay_body = json.loads(retry_writer.data.split(b"\r\n\r\n", 1)[1])
    assert replay_body["ack_seq"] == 1
    await asyncio.wait_for(bus.consume_inbound(), timeout=1)
    assert len(channel.calls) == 1
    await ingress.close()


@pytest.mark.asyncio
async def test_camera_synthetic_waits_for_interleaved_marina_turn(tmp_path: Path) -> None:
    bus = MessageBus()
    ordinary_started = asyncio.Event()
    ordinary_release = asyncio.Event()
    camera_started = asyncio.Event()

    class Runtime:
        async def stream_message(self, message, session_key):
            assert session_key == "telegram:123"
            if message.sender_id == "123":
                ordinary_started.set()
                await ordinary_release.wait()
            else:
                camera_started.set()
            yield GatewayStreamUpdate(kind="final", text="done")

    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=Runtime(), workspace=tmp_path)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="123",
                chat_id="123",
                content="ordinary",
                metadata={"is_group": False},
            )
        )
        await asyncio.wait_for(ordinary_started.wait(), timeout=1)
        await bus.publish_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="__camera__",
                chat_id="123",
                content="photo",
                metadata={
                    "_synthetic": True,
                    "_camera_authority": CAMERA_AUTHORITY,
                    "_camera_candidate_id": "candidate",
                },
            )
        )
        await asyncio.sleep(0.05)
        assert not camera_started.is_set()
        assert not bridge._session_tasks["telegram:123"].cancelled()
        ordinary_release.set()
        await asyncio.wait_for(camera_started.wait(), timeout=1)
    finally:
        bridge.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_bound_answer_uses_validated_durable_honcho_path_only(tmp_path: Path) -> None:
    config = GatewayConfig(
        enabled_channels=["telegram"],
        conversation_learning=True,
        evals_capture=True,
        memory_backend="shadow",
        honcho_base_url="https://honcho.test",
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={"marina": {"workspace": "w", "api_key": "a", "observed_peer": "p"}},
        camera_ingress=CameraIngressConfig(
            enabled=True,
            listen_port=8765,
            bearer_token_file=tmp_path / "token",
            principal="123",
            tenant_id="marina",
            chat_id="123",
            session_key="telegram:123",
        ),
    )
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = config
    pool._session_owner_principals = {"session": "123"}
    calls = []

    async def append_exchange(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(assistant_message_id="honcho-1", assistant_client_op_id="op-1")

    pool._shadow_backend_for_scope = lambda scope: SimpleNamespace(append_exchange=append_exchange)
    scope = MemoryScope(private_tenant="marina", shared_tenants=())
    synthetic_ctx = TurnContext(
        principal="__camera__",
        is_owner=False,
        is_private=True,
        channel="telegram",
        chat_id="123",
        session_id="session",
        camera_authorized=True,
    )
    synthetic = InboundMessage(
        channel="telegram",
        sender_id="__camera__",
        chat_id="123",
        content="analyze",
        metadata={"_camera_authority": CAMERA_AUTHORITY},
    )
    assert (
        await pool._append_conversation_turn(
            turn_ctx=synthetic_ctx,
            memory_scope=scope,
            message=synthetic,
            user_text="analyze",
            assistant_text="question",
        )
        is None
    )
    assert calls == []

    answer_ctx = TurnContext(
        principal="123",
        is_owner=False,
        is_private=True,
        channel="telegram",
        chat_id="123",
        session_id="session",
        camera_authorized=True,
    )
    answer = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Я это съела",
        metadata={
            "is_group": False,
            "message_id": 90,
            "reply_to_message_id": 77,
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_answer": "yes",
            "_camera_candidate_id": "dropbox-camera-v1-" + "a" * 64,
        },
    )
    recorder = SimpleNamespace(
        validated_nutrition_envelope={
            "schema_version": 2,
            "record_type": "meal_observation",
            "basis": ["image"],
            "consumption_status": "consumed",
            "energy_kcal_best": 200,
        },
        decision_trace_status="recorded",
        nutrition_annotation_status="recorded",
        decision_trace_envelope=None,
        episode_id="offline",
    )
    receipt = await pool._append_conversation_turn(
        turn_ctx=answer_ctx,
        memory_scope=scope,
        message=answer,
        recorder=recorder,
        user_text=answer.content,
        assistant_text="Записано",
    )
    assert receipt.assistant_message_id == "honcho-1"
    assert calls[0][1]["durable"] is True
    assert (
        calls[0][1]["assistant_metadata"]["camera_candidate_id"]
        == answer.metadata["_camera_candidate_id"]
    )
    assert calls[0][1]["assistant_metadata"]["camera_answer_bound"] == "yes"
    recorder.validated_nutrition_envelope = {
        **recorder.validated_nutrition_envelope,
        "consumption_status": "unknown",
    }
    with pytest.raises(ValueError, match="consumed image observation"):
        await pool._append_conversation_turn(
            turn_ctx=answer_ctx,
            memory_scope=scope,
            message=answer,
            recorder=recorder,
            user_text=answer.content,
            assistant_text="not recorded",
        )
    assert len(calls) == 1
