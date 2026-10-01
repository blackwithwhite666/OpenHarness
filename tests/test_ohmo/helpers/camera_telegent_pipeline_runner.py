#!/usr/bin/env python3
"""Run the real Telegent Camera pipeline against the synthetic Ohmo TCP fixture."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from email import policy
from email.parser import BytesParser
import hashlib
import httpx
from io import BytesIO
import json
import os
from pathlib import Path
import sys


def _parts(request: httpx.Request) -> dict[str, bytes]:
    envelope = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in envelope.iter_parts()
    }


class LoopbackTransport(httpx.BaseTransport):
    """Forward the real serialized client request to an HTTP loopback listener."""

    def __init__(
        self, host: str, port: int, *, drop_seq: int | None = None,
        drop_before_forward_seq: int | None = None,
    ):
        self.host = host
        self.port = port
        self.drop_seq = drop_seq
        self.drop_before_forward_seq = drop_before_forward_seq
        self.dropped = False
        self.client = httpx.Client(trust_env=False)
        self.observed: list[dict[str, object]] = []
        self.source_operations: list[dict[str, object]] = []
        self.reference_queries: list[dict[str, object]] = []
        self.operation_order: list[str] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        self.operation_order.append(request.url.path)
        headers = request.headers.copy()
        headers["Host"] = f"{self.host}:{self.port}"
        forwarded = httpx.Request(
            method=request.method,
            url=request.url.copy_with(scheme="http", host=self.host, port=self.port),
            headers=headers,
            content=body,
            extensions=request.extensions,
        )
        is_candidate = request.url.path in {
            "/internal/v1/camera/candidates", "/internal/v1/camera/reconcile"
        }
        source_observed = None
        reference_observed = None
        if request.url.path == "/internal/v1/camera/references":
            reference_observed = {"request": json.loads(body)}
            self.reference_queries.append(reference_observed)
        if request.url.path == "/internal/v1/camera/reference-source":
            parts = _parts(request)
            envelope = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + body
            )
            image_part = next(
                part for part in envelope.iter_parts()
                if part.get_param("name", header="content-disposition") == "image"
            )
            self.source_operations.append({
                "method": request.method,
                "path": request.url.path,
                "request": json.loads(parts["request"]),
                "manifest_sha256": hashlib.sha256(parts["manifest"]).hexdigest(),
                "producer_sha256": hashlib.sha256(parts["producer"]).hexdigest(),
                "image_sha256": hashlib.sha256(parts["image"]).hexdigest(),
                "image_filename": image_part.get_filename(),
                "image_content_type": image_part.get_content_type(),
            })
            source_observed = self.source_operations[-1]
        observed = None
        if is_candidate:
            parts = _parts(request)
            candidate = json.loads(parts["request"])
            observed = {
                "method": request.method,
                "path": request.url.path,
                "purpose": request.headers.get("x-camera-reconcile-purpose"),
                "request": candidate,
                "content_type": request.headers["content-type"],
                "content_length": request.headers.get("content-length"),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "image_sha256": hashlib.sha256(parts["image"]).hexdigest(),
                "response_status": None,
                "response_body": None,
            }
            self.observed.append(observed)
            if candidate["seq"] == self.drop_before_forward_seq and not self.dropped:
                self.dropped = True
                raise httpx.ConnectError("synthetic pre-send network loss")
        response = self.client.send(forwarded)
        if source_observed is not None:
            source_observed["response_status"] = response.status_code
            source_observed["response_body"] = response.content.decode("utf-8", "replace")
        if reference_observed is not None:
            reference_observed["response_status"] = response.status_code
            reference_observed["response_body"] = response.content.decode("utf-8", "replace")
        if observed is not None:
            observed["response_status"] = response.status_code
            observed["response_body"] = response.content.decode("utf-8", "replace")
            candidate = observed["request"]
            if candidate["seq"] == self.drop_seq and not self.dropped:
                self.dropped = True
                response.close()
                raise httpx.ReadError("synthetic lost ACK after Ohmo committed request")
        return response

    def close(self) -> None:
        self.client.close()


def _pipeline(
    checkout: Path, state_dir: Path, port: int, token: str, *,
    drop_seq=None, drop_before_forward_seq=None, scene_verdict="non_food",
    include_repeat=False,
):
    sys.path.insert(0, str(checkout))
    from telegent.health_advisor.dropbox_camera.camera_submission import CameraSubmissionClient
    from telegent.health_advisor.dropbox_camera.clip_prefilter import ClipFoodPrefilter
    from telegent.health_advisor.dropbox_camera.pipeline import DropboxCameraPipeline
    from telegent.health_advisor.dropbox_camera.camera_scene_comparator import TrustedInputPolicy
    from telegent.health_advisor.dropbox_camera.camera_scene_gate import SceneGate
    from telegent.health_advisor.dropbox_camera.camera_scene_producer import CameraSceneProducerGate
    from telegent.health_advisor.dropbox_camera.store import CameraArtifactStore
    from telegent.health_advisor.dropbox_camera._models_classifier import ActiveClassifierRelease
    from telegent.health_advisor.dropbox_camera.deepseek_runtime import load_production_release
    from telegent.health_advisor.test_dropbox_camera_pipeline import (
        _Source, _Classifier, _entry, _clip_release,
    )
    from telegent.health_advisor.dropbox_camera.test_camera_scene_comparator import _runtime
    from PIL import Image, ImageDraw

    release_record = json.loads(
        (checkout / "telegent/data/dropbox_camera_deepseek_production_release.json").read_text()
    )
    release = ActiveClassifierRelease.model_validate(release_record["release"])
    scene_release = load_production_release()
    clock_path = state_dir / "capture-clock.json"
    if clock_path.exists():
        now = datetime.fromisoformat(json.loads(clock_path.read_text())["now"])
    else:
        now = datetime.now(UTC).replace(microsecond=0)
        clock_path.write_text(json.dumps({"now": now.isoformat()}), encoding="utf-8")
    older = now - timedelta(minutes=2)
    newer = now - timedelta(minutes=1)
    newest = now
    source_root = state_dir / "generated"
    source_root.mkdir(exist_ok=True)
    image_a_path, image_b_path, image_c_path = (
        source_root / "a.jpg", source_root / "b.jpg", source_root / "c.jpg"
    )
    if image_a_path.exists() and image_b_path.exists() and image_c_path.exists():
        image_a, image_b, image_c = (
            image_a_path.read_bytes(), image_b_path.read_bytes(), image_c_path.read_bytes()
        )
    else:
        def synthetic_image(captured: datetime, *, vertical: bool) -> bytes:
            image = Image.new("RGB", (96, 64), (24, 34, 42))
            draw = ImageDraw.Draw(image)
            if vertical:
                for x in range(0, 96, 12):
                    draw.rectangle((x, 0, x + 5, 63), fill=(220, 180, 50))
            else:
                for y in range(0, 64, 10):
                    draw.rectangle((0, y, 95, y + 4), fill=(40, 190, 130))
            exif = image.getexif()
            exif[36867] = captured.strftime("%Y:%m:%d %H:%M:%S")
            exif[36881] = "+00:00"
            output = BytesIO()
            image.save(output, format="JPEG", exif=exif)
            return output.getvalue()

        image_a = synthetic_image(older, vertical=False)
        image_a_path.write_bytes(image_a)
        image_b = synthetic_image(newer, vertical=True)
        image_b_path.write_bytes(image_b)
        image_c_image = Image.new("RGB", (96, 64), (76, 35, 145))
        image_c_draw = ImageDraw.Draw(image_c_image)
        for y in range(0, 64, 8):
            image_c_draw.line((0, y, 95, 63 - y), fill=(35, 210, 230), width=3)
        image_c_exif = image_c_image.getexif()
        image_c_exif[36867] = newest.strftime("%Y:%m:%d %H:%M:%S")
        image_c_exif[36881] = "+00:00"
        image_c_output = BytesIO()
        image_c_image.save(image_c_output, format="JPEG", exif=image_c_exif)
        image_c = image_c_output.getvalue()
        image_c_path.write_bytes(image_c)
    entries = [
        _entry("id:tcp-older", "rev:tcp-a").model_copy(update={
            "server_modified": older, "size": len(image_a), "name": "a.jpg",
            "path_lower": "/camera/a.jpg", "path_display": "/camera/a.jpg",
        }),
        _entry("id:tcp-newer", "rev:tcp-b").model_copy(update={
            "server_modified": newer, "size": len(image_b), "name": "b.jpg",
            "path_lower": "/camera/b.jpg", "path_display": "/camera/b.jpg",
        }),
        _entry("id:tcp-repeat", "rev:tcp-c").model_copy(update={
            "server_modified": newest, "size": len(image_c), "name": "c.jpg",
            "path_lower": "/camera/c.jpg", "path_display": "/camera/c.jpg",
        }),
    ]
    if not include_repeat:
        entries = entries[:2]
    source = _Source(entries, image_b)
    original_download = source.download_revision

    def download(entry, *, destination_dir):
        source.payload = {
            "id:tcp-older": image_a,
            "id:tcp-newer": image_b,
            "id:tcp-repeat": image_c,
        }[entry.file_id]
        return original_download(entry, destination_dir=destination_dir)

    source.download_revision = download
    store = CameraArtifactStore(state_dir / "camera-artifacts.db")
    transport = LoopbackTransport(
        "127.0.0.1", port, drop_seq=drop_seq,
        drop_before_forward_seq=drop_before_forward_seq,
    )
    client = httpx.Client(transport=transport, trust_env=False, timeout=3)
    submission = CameraSubmissionClient(
        store=store,
        base_url="https://camera.private",
        private_https_origins={"https://camera.private"},
        service_token=token,
        timeout=3,
        http_client=client,
        clock=lambda: now,
    )
    scene_root = state_dir / "scene-inputs"
    scene_root.mkdir(mode=0o700, exist_ok=True)
    trusted_inputs = TrustedInputPolicy(scene_root, owner_uid=os.getuid())
    _scene_source, scene_quote, scene_classifier, _scene_requests = _runtime(
        state_dir / "scene-adapter",
        decision=scene_verdict,
        source_override=scene_release,
        quote_override=scene_release.quote,
    )

    def make_scene_gate(scope):
        return SceneGate(
            release=scene_release,
            quote=scene_quote,
            classifier=scene_classifier,
            trusted_inputs=trusted_inputs,
            scope=scope,
            clock=lambda: now,
        )

    scene_producer = CameraSceneProducerGate(
        gate_factory=make_scene_gate,
        scene_classifier=scene_classifier,
        client=submission,
        store=store,
        source=source,
        trusted_inputs=trusted_inputs,
        input_root=scene_root,
    )
    scene_producer._fixture_scene_requests = _scene_requests
    pipeline = DropboxCameraPipeline(
        source=source,
        classifier=_Classifier(
            model=release.model,
            prompt_version=release.prompt_version,
            policy_version=release.policy_version,
        ),
        release=release,
        source_path="/camera",
        store=store,
        max_attempts=3,
        retry_base_ms=10,
        retry_max_ms=100,
        camera_submission_client=submission,
        scene_gate=scene_producer,
        clip_prefilter=ClipFoodPrefilter(_clip_release(), scorer=lambda *_: 0.0),
        clock=lambda: now,
    )
    pipeline._fixture_scene_producer = scene_producer
    return pipeline, store, transport, entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegent-checkout", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token", required=True)
    parser.add_argument("--phase", choices={"initial", "reopen"}, required=True)
    parser.add_argument("--scenario", choices={"lost_ack_replay", "negative_expiry", "scene_source_repair"},
                        default="lost_ack_replay")
    parser.add_argument("--stage", choices={"initial", "distinct", "repeat"}, default="initial")
    parser.add_argument("--scene-verdict", choices={"non_food", "food", "ambiguous"},
                        default="non_food")
    args = parser.parse_args()
    checkout = args.telegent_checkout.resolve(strict=True)
    if not checkout.is_dir() or checkout.is_symlink():
        raise SystemExit("TELEGENT_CHECKOUT must name a checkout directory")
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    if args.scenario == "scene_source_repair":
        pipeline, store, transport, entries = _pipeline(
            checkout, args.state_dir, args.port, args.token,
            scene_verdict=args.scene_verdict,
            include_repeat=True,
        )
        selected = {
            "initial": entries[:1],
            "distinct": entries[:2],
            "repeat": entries[2:],
        }[args.stage]
        pipeline._source.entries = selected
        summary = pipeline.run_once()
        stage_results = list(summary.results)
        target = selected[-1]
        from telegent.health_advisor.dropbox_camera._models_base import candidate_id_for
        target_id = candidate_id_for(target.file_id, target.rev)
        target_result = next(
            (item for item in stage_results if item.candidate_id == target_id), None
        )
        if target_result is None or target_result.outcome == "published":
            manifest = store.load_manifest(target_id)
            sidecar = store.load_producer(target_id)
            if manifest is None or sidecar is None:
                raise SystemExit("fixture target was not durably published")
            image_path = args.state_dir / "generated" / target.name
            stage_results.append(
                pipeline._submit_published_candidate(sidecar, image_path.read_bytes())
            )
        state = store.load_submission_state()
        scene_gate = pipeline._fixture_scene_producer
        report = {
            "phase": args.stage,
            "target_candidate_id": target_id,
            "outcomes": [item.outcome for item in stage_results],
            "results": [{"candidate_id": item.candidate_id, "outcome": item.outcome,
                         "submission_outcome": item.submission_outcome,
                         "error_class": item.error_class,
                         "scene_source_operations": item.scene_source_operations,
                         "scene_sources_reestablished": item.scene_sources_reestablished,
                         "scene_reference_hold_count": item.scene_reference_hold_count}
                        for item in stage_results],
            "observed": transport.observed,
            "source_operations": transport.source_operations,
            "reference_queries": transport.reference_queries,
            "operation_order": transport.operation_order,
            "publication_artifacts": {
                entry.file_id: {
                    "candidate_id": candidate_id_for(entry.file_id, entry.rev),
                    "manifest_sha256": hashlib.sha256(
                        store.load_manifest_bytes(candidate_id_for(entry.file_id, entry.rev))
                    ).hexdigest(),
                    "sidecar_sha256": hashlib.sha256(
                        store.load_producer_bytes(candidate_id_for(entry.file_id, entry.rev))
                    ).hexdigest(),
                    "original_filename": store.load_manifest(
                        candidate_id_for(entry.file_id, entry.rev)
                    ).original_filename,
                    "mime_type": store.load_manifest(
                        candidate_id_for(entry.file_id, entry.rev)
                    ).mime_type,
                }
                for entry in entries
                if store.load_manifest_bytes(candidate_id_for(entry.file_id, entry.rev))
                is not None
                and store.load_producer_bytes(candidate_id_for(entry.file_id, entry.rev))
                is not None
            },
            "committed_seq": state.get("committed_seq") if state else None,
            "scene_source_operations": scene_gate.source_operation_calls,
            "scene_sources_reestablished": scene_gate.source_reestablished_count,
            "scene_reference_hold_count": scene_gate.reference_hold_count,
            "scene_provider_requests": len(scene_gate._fixture_scene_requests),
        }
        pipeline.close()
        transport.close()
        print(json.dumps(report, separators=(",", ":")), flush=True)
        return

    pipeline, store, transport, entries = _pipeline(
        checkout, args.state_dir, args.port, args.token,
        drop_seq=1 if args.phase == "initial" and args.scenario == "lost_ack_replay" else None,
        drop_before_forward_seq=(
            1 if args.phase == "initial" and args.scenario == "negative_expiry" else None
        ),
        include_repeat=args.scenario == "scene_source_repair",
    )
    recovery = None
    if args.phase == "initial":
        summary = pipeline.run_once()
        state = store.load_submission_state()
        active = state.get("active_attempt") if state else None
        frozen = active.get("request") if isinstance(active, dict) else None
        report = {
            "phase": "initial",
            "candidate_fifo": [item["request"]["candidate_id"] for item in transport.observed],
            "sequences": [item["request"]["seq"] for item in transport.observed],
            "outcomes": [item.outcome for item in summary.results],
            "results": [{"candidate_id": item.candidate_id, "outcome": item.outcome,
                         "error_class": item.error_class,
                         "submission_outcome": item.submission_outcome}
                        for item in summary.results],
            "active_request": frozen,
            "observed": transport.observed,
            "candidate_ids": [entry.file_id for entry in entries],
            "drop_occurred": transport.dropped,
        }
        pipeline.close()
        transport.close()
        print(json.dumps(report, separators=(",", ":")), flush=True)
        command = json.loads(sys.stdin.buffer.readline(512))
        if not isinstance(command, dict) or command.get("op") != "reopen":
            raise SystemExit("expected reopen phase control")
        port = int(command["port"])
        pipeline, store, transport, entries = _pipeline(
            checkout, args.state_dir, port, args.token,
            include_repeat=args.scenario == "scene_source_repair",
        )
    state = store.load_submission_state()
    active = state.get("active_attempt") if state else None
    if not isinstance(active, dict):
        raise SystemExit("producer reopened without its durable active request")
    if args.scenario == "lost_ack_replay":
        recovery = pipeline._camera_submission_client.reconcile(
            active["candidate_id"],
            (args.state_dir / "generated" / "b.jpg").read_bytes(),
            purpose="existing_outcome_only",
        )
        summary = pipeline.run_once()
    else:
        summary = pipeline.run_once()
    state = store.load_submission_state()
    active = state.get("active_attempt") if state else None
    from telegent.health_advisor.dropbox_camera._models_base import candidate_id_for

    report = {
        "phase": "reopen",
        "recovery_outcome": recovery.outcome if recovery is not None else None,
        "outcomes": [item.outcome for item in summary.results],
        "results": [{"candidate_id": item.candidate_id, "outcome": item.outcome,
                     "error_class": item.error_class,
                     "submission_outcome": item.submission_outcome}
                    for item in summary.results],
        "observed": transport.observed,
        "state_committed_seq": state.get("committed_seq") if state else None,
        "lease": state.get("lease") if state else None,
        "active_request": active,
        "attempt_results": {
            candidate_id_for(entry.file_id, entry.rev): store.load_submission_attempt(
                candidate_id_for(entry.file_id, entry.rev)
            ) for entry in entries
        },
    }
    pipeline.close()
    transport.close()
    print(json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
