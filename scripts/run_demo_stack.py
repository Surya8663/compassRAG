import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

SERVICES = [
    {
        "name": "Ingestion Service",
        "pkg": "compass-rag-ingestion",
        "dir": "services/ingestion",
        "port": 8001,
    },
    {
        "name": "Retrieval Service",
        "pkg": "compass-rag-retrieval",
        "dir": "services/retrieval",
        "port": 8002,
    },
    {
        "name": "Correction Service",
        "pkg": "compass-rag-correction",
        "dir": "services/correction",
        "port": 8003,
    },
    {
        "name": "Generation Service",
        "pkg": "compass-rag-generation",
        "dir": "services/generation",
        "port": 8004,
    },
    {
        "name": "API Gateway",
        "pkg": "compass-rag-api-gateway",
        "dir": "services/api-gateway",
        "port": 8000,
    },
]


async def wait_for_health(
    client: httpx.AsyncClient, name: str, url: str, timeout: float = 60.0
) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = await client.get(url, timeout=2.0, follow_redirects=True)
            if resp.status_code in (200, 307, 308):
                print(f"[OK] {name} is live and healthy ({url})")
                return True
        except Exception:
            await asyncio.sleep(0.5)
    print(f"[FAIL] {name} failed to return 200 within {timeout}s at {url}")
    return False


async def main() -> int:
    env = os.environ.copy()
    root_dir = str(Path(__file__).resolve().parent.parent)
    env.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "WARNING",
        "JSON_LOGS": "False",
        "POSTGRES_DSN": "postgresql+asyncpg://postgres:postgres@localhost:5432/compass_rag",
        "REDIS_URL": "redis://localhost:6379/0",
        "QDRANT_URL": "http://localhost:6333",
        "ELASTICSEARCH_URL": "http://localhost:9200",
        "EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
        "EMBEDDING_DIMENSION": "384",
        "LLM_MODEL_NAME": "gemini-3.5-flash",
        "SIMILARITY_THRESHOLD": "0.75",
        "RETRIEVAL_TOP_K": "10",
        "CORRECTION_CONFIDENCE_THRESHOLD": "0.80",
        "MAX_RETRIES": "3",
        "INGESTION_SERVICE_URL": "http://localhost:8001",
        "RETRIEVAL_SERVICE_URL": "http://localhost:8002",
        "CORRECTION_SERVICE_URL": "http://localhost:8003",
        "GENERATION_SERVICE_URL": "http://localhost:8004",
        "PYTHONPATH": root_dir,
    })

    processes = []
    log_files = []
    print("====================================================================")
    print("STARTING COMPASS RAG FULL STACK (BACKEND SERVICES + FRONTEND)")
    print("====================================================================")

    uv_bin = shutil.which("uv") or r"C:\Users\surya\AppData\Roaming\Python\Python312\Scripts\uv.exe"
    logs_dir = Path(root_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    print(f"Service logs are being written to: {logs_dir}")

    try:
        # Start all 5 backend microservices
        for s in SERVICES:
            cmd = [
                uv_bin,
                "run",
                "--package",
                str(s["pkg"]),
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(s["port"]),
            ]
            print(f"Starting {s['name']} on port {s['port']}...")
            log_f = open(logs_dir / f"{str(s['name']).lower().replace(' ', '_')}.log", "w", encoding="utf-8")
            log_files.append(log_f)
            p = subprocess.Popen(
                cmd,
                cwd=str(Path(root_dir) / str(s["dir"])),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            processes.append((s["name"], p))

        # Start Next.js Frontend
        print("Starting Next.js Enterprise Frontend on port 3000...")
        node_bin = shutil.which("node.exe") or shutil.which("node") or "node"
        frontend_cmd = [node_bin, "node_modules/next/dist/bin/next", "start", "-p", "3000"]
        log_f_frontend = open(logs_dir / "nextjs_frontend.log", "w", encoding="utf-8")
        log_files.append(log_f_frontend)
        p_frontend = subprocess.Popen(
            frontend_cmd,
            cwd=str(Path(root_dir) / "frontend"),
            env=env,
            stdout=log_f_frontend,
            stderr=subprocess.STDOUT,
        )
        processes.append(("Next.js Frontend", p_frontend))

        async with httpx.AsyncClient() as client:
            # Wait for backend services
            backend_tasks = [
                wait_for_health(client, str(s["name"]), f"http://localhost:{s['port']}/health")
                for s in SERVICES
            ]
            backend_results = await asyncio.gather(*backend_tasks)

            # Wait for frontend
            frontend_live = await wait_for_health(client, "Next.js Frontend", "http://localhost:3000")

        success_count = sum(1 for r in backend_results if r) + (1 if frontend_live else 0)
        total_count = len(SERVICES) + 1
        print("====================================================================")
        print(f"Stack Status: {success_count}/{total_count} services active and responding.")
        print("====================================================================")
        print("Frontend UI available at: http://localhost:3000")
        print("API Gateway available at: http://localhost:8000")
        print("Keep this process running to use the application. Press Ctrl+C to stop.")

        # Keep alive indefinitely until terminated
        while True:
            await asyncio.sleep(5)
            # Check if any process died
            for name, p in processes:
                if p.poll() is not None:
                    print(f"[WARN] Process '{name}' exited unexpectedly with code {p.poll()}")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nShutting down all local stack services...")
    finally:
        for _name, p in processes:
            p.terminate()
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
