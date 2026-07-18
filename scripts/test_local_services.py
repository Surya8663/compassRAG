import asyncio
import os
import subprocess
import sys
import time
import httpx

SERVICES = [
    {"name": "Ingestion Service", "pkg": "compass-rag-ingestion", "dir": "services/ingestion", "port": 8001},
    {"name": "Retrieval Service", "pkg": "compass-rag-retrieval", "dir": "services/retrieval", "port": 8002},
    {"name": "Correction Service", "pkg": "compass-rag-correction", "dir": "services/correction", "port": 8003},
    {"name": "Generation Service", "pkg": "compass-rag-generation", "dir": "services/generation", "port": 8004},
    {"name": "API Gateway", "pkg": "compass-rag-api-gateway", "dir": "services/api-gateway", "port": 8000},
]

async def wait_for_health(client: httpx.AsyncClient, name: str, port: int, timeout: float = 15.0) -> bool:
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = await client.get(url, timeout=2.0)
            if resp.status_code == 200:
                print(f"[OK] {name} (port {port}) health check passed: {resp.json()}")
                return True
        except Exception:
            await asyncio.sleep(0.5)
    print(f"[FAIL] {name} (port {port}) failed to return 200 within {timeout}s")
    return False

async def main() -> int:
    # Ensure environment variables are loaded
    env = os.environ.copy()
    env.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "WARNING",
        "JSON_LOGS": "False",
        "POSTGRES_DSN": "postgresql+asyncpg://postgres:postgres@localhost:5432/compass_rag",
        "REDIS_URL": "redis://localhost:6379/0",
        "QDRANT_URL": "http://localhost:6333",
        "ELASTICSEARCH_URL": "http://localhost:9200",
        "EMBEDDING_MODEL_NAME": "BAAI/bge-large-en-v1.5",
        "EMBEDDING_DIMENSION": "1024",
        "LLM_MODEL_NAME": "gpt-4o-mini",
        "SIMILARITY_THRESHOLD": "0.75",
        "RETRIEVAL_TOP_K": "10",
        "CORRECTION_CONFIDENCE_THRESHOLD": "0.80",
        "MAX_RETRIES": "3",
        "INGESTION_SERVICE_URL": "http://localhost:8001",
        "RETRIEVAL_SERVICE_URL": "http://localhost:8002",
        "CORRECTION_SERVICE_URL": "http://localhost:8003",
        "GENERATION_SERVICE_URL": "http://localhost:8004",
    })

    processes = []
    print("====================================================================")
    print("STARTING ALL 5 FASTAPI SERVICES LOCALLY FOR VERIFICATION")
    print("====================================================================")
    
    try:
        for s in SERVICES:
            cmd = ["python", "-m", "uv", "run", "--package", s["pkg"], "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(s["port"])]
            print(f"Starting {s['name']} on port {s['port']}...")
            p = subprocess.Popen(cmd, cwd=s["dir"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append((s["name"], p))
        
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(*(wait_for_health(client, s["name"], s["port"]) for s in SERVICES))
        
        success_count = sum(1 for r in results if r)
        total_count = len(SERVICES)
        print("====================================================================")
        print(f"Verification Summary: {success_count}/{total_count} services healthy.")
        return 0 if success_count == total_count else 1
    finally:
        print("Stopping all local service processes...")
        for name, p in processes:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
