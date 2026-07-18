import asyncio
import sys

import httpx

SERVICES = [
    {"name": "API Gateway", "url": "http://localhost:8000/health"},
    {"name": "Ingestion Service", "url": "http://localhost:8001/health"},
    {"name": "Retrieval Service", "url": "http://localhost:8002/health"},
    {"name": "Correction Service", "url": "http://localhost:8003/health"},
    {"name": "Generation Service", "url": "http://localhost:8004/health"},
]


async def check_service_health(client: httpx.AsyncClient, service: dict[str, str]) -> bool:
    name = service["name"]
    url = service["url"]
    try:
        response = await client.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] {name} is healthy: {data}")
            return True
        else:
            print(f"[FAIL] {name} returned status code {response.status_code}: {response.text}")
            return False
    except Exception as exc:
        print(f"[FAIL] {name} unreachable at {url} -> {exc}")
        return False


async def main() -> int:
    print("====================================================================")
    print("VERIFYING COMPASS RAG SERVICES HEALTH CHECKS")
    print("====================================================================")
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(check_service_health(client, s) for s in SERVICES))

    success_count = sum(1 for r in results if r)
    total_count = len(SERVICES)
    print("====================================================================")
    print(f"Summary: {success_count}/{total_count} services healthy.")

    if success_count == total_count:
        print("[SUCCESS] All Compass RAG services are running and healthy.")
        return 0
    else:
        print("[ERROR] One or more services failed health check verification.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
