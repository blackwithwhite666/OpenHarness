"""Verify a real Honcho v3 message commit against the isolated Compose API."""

from __future__ import annotations

import asyncio
import os

from ohmo.memory_service.honcho_client import HonchoClient


async def main() -> None:
    url = os.environ["CAMERA_HONCHO_URL"]
    workspace = "camera-probe"
    session = "camera-probe-session"
    operation = "camera-probe-synthetic-v1"
    async with HonchoClient(url, "local-auth-disabled", workspace) as client:
        await client.get_or_create_workspace()
        await client.get_or_create_peer("ohmo")
        await client.get_or_create_peer("owner")
        await client.get_or_create_session(session, peers={"ohmo": {}, "owner": {}})
        existing = await client.find_messages_by_client_op_id(session, operation)
        if not existing:
            created = await client.create_messages(
                session,
                [
                    {
                        "content": "Synthetic Camera probe observation",
                        "peer_id": "ohmo",
                        "metadata": {
                            "client_op_id": operation,
                            "event_type": "camera_probe",
                            "synthetic": True,
                        },
                    }
                ],
            )
            assert len(created) == 1
        found = await client.find_messages_by_client_op_id(session, operation)
        assert len(found) == 1
        assert found[0].metadata["event_type"] == "camera_probe"
        print(f"PASS Honcho v3 message commit and exact readback: id={found[0].id}")


if __name__ == "__main__":
    asyncio.run(main())
