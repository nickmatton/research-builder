"""Thin async client for the Lambda Cloud API.

Docs: https://docs.lambda.ai/on-demand-cloud/cloud-api/

We use these operations:
  - launch instance     (POST /instance-operations/launch)
  - get instance        (GET  /instances/{id})
  - terminate instance  (POST /instance-operations/terminate)
  - add SSH key         (POST /ssh-keys)

Authentication is via the ``Authorization: Bearer`` header sourced from
LAMBDA_API_KEY.

Per phase attempt we generate an ephemeral ed25519 keypair (via the system
``ssh-keygen`` binary) and register it with the Lambda API before launching.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

LAMBDA_BASE_URL = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_SSH_USER = "ubuntu"


@dataclass
class LambdaInstance:
    id: str
    public_ip: str
    ssh_user: str
    ssh_key_path: Path  # private key on disk
    instance_type: str
    state: str


def generate_ssh_keypair(target_dir: Path) -> tuple[Path, str]:
    """Generate an ephemeral ed25519 keypair under target_dir.

    Returns (private_key_path, public_key_text). Uses the system ``ssh-keygen``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    priv = target_dir / "id_ed25519"
    pub = target_dir / "id_ed25519.pub"
    if priv.exists():
        priv.unlink()
    if pub.exists():
        pub.unlink()
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(priv), "-C", "research-builder"],
        check=True,
    )
    priv.chmod(0o600)
    return priv, pub.read_text().strip()


class LambdaClient:
    """Minimal async wrapper around the Lambda Cloud API."""

    def __init__(self, api_key: str, *, base_url: str = LAMBDA_BASE_URL) -> None:
        if not api_key:
            raise ValueError("LAMBDA_API_KEY is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def add_ssh_key(self, name: str, public_key: str) -> str:
        """Register an SSH key with Lambda Cloud. Returns the key name.

        If a key with the same name already exists, returns the existing name
        without error.
        """
        body = {"name": name, "public_key": public_key}
        logger.info("Lambda add_ssh_key: name=%s", name)
        resp = await self._client.post("/ssh-keys", json=body)
        # Lambda returns 400 with "Key with name already exists" on duplicate
        if resp.status_code == 400:
            data = resp.json()
            err_msg = str(data.get("error", {}).get("message", ""))
            if "already exists" in err_msg.lower():
                logger.debug("SSH key '%s' already registered, reusing", name)
                return name
        resp.raise_for_status()
        return name

    async def launch_instance(
        self,
        *,
        instance_type: str,
        ssh_key_names: list[str],
        name: str,
        region: str,
    ) -> LambdaInstance:
        body = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
            "name": name,
        }
        logger.info("Lambda launch_instance: type=%s region=%s name=%s", instance_type, region, name)
        resp = await self._client.post("/instance-operations/launch", json=body)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        instance_ids = data.get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError(f"Lambda launch returned no instance IDs: {resp.json()}")
        return LambdaInstance(
            id=instance_ids[0],
            public_ip="",
            ssh_user=DEFAULT_SSH_USER,
            ssh_key_path=Path(),  # filled in by caller
            instance_type=instance_type,
            state="booting",
        )

    async def get_instance(self, instance_id: str) -> dict:
        resp = await self._client.get(f"/instances/{instance_id}")
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def wait_until_ready(
        self, instance_id: str, *, timeout_s: int = 600, poll_interval_s: float = 10.0
    ) -> dict:
        """Poll until the instance reports status=='active' and has a public IP."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        last: dict = {}
        while asyncio.get_event_loop().time() < deadline:
            last = await self.get_instance(instance_id)
            status = last.get("status")
            ip = last.get("ip")
            logger.debug("Lambda instance=%s status=%s ip=%s", instance_id, status, ip)
            if status == "active" and ip:
                return last
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"Lambda instance {instance_id} did not become active within {timeout_s}s "
            f"(last status={last.get('status')!r})"
        )

    async def get_instance_type_price(self, instance_type: str) -> float | None:
        """Fetch the hourly price in USD for an instance type, or None if unavailable."""
        try:
            resp = await self._client.get("/instance-types")
            resp.raise_for_status()
            data = resp.json().get("data", {})
            type_info = data.get(instance_type)
            if type_info:
                price = type_info.get("instance_type", {}).get("price_cents_per_hour")
                if price is not None:
                    return price / 100.0
        except Exception:
            logger.debug("Failed to fetch instance type pricing", exc_info=True)
        return None

    async def terminate_instance(self, instance_id: str) -> None:
        logger.info("Lambda terminate_instance: id=%s", instance_id)
        body = {"instance_ids": [instance_id]}
        resp = await self._client.post("/instance-operations/terminate", json=body)
        # Treat 404 as already-gone — terminate is idempotent from our POV.
        if resp.status_code == 404:
            return
        resp.raise_for_status()
