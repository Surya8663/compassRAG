import httpx
import json
import asyncio
from pathlib import Path

async def main():
    print("==========================================================")
    print("STEP 1: UPLOADING RESUME TO API GATEWAY (/v1/ingest)")
    print("==========================================================")
    
    pdf_path = Path("C:/Users/surya/CompassRAG/services/ingestion/.staging/c9e73fe6-e13c-4ab3-903e-b604a7d86340_SURYA_RES_UME (5).pdf")
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(pdf_path, "rb") as f:
            files = {"file": ("Surya_Resume.pdf", f.read(), "application/pdf")}
            data = {"document_id": "surya_resume_2026", "tenant_id": "tenant_enterprise"}
            
            print("Sending POST /v1/ingest to http://localhost:8000...")
            resp = await client.post(
                "http://localhost:8000/v1/ingest",
                files=files,
                data=data,
                headers={"X-Tenant-ID": "tenant_enterprise"}
            )
            print(f"Upload Status Code: {resp.status_code}")
            print(f"Upload Response: {resp.text}\n")

        print("==========================================================")
        print("STEP 2: QUERYING THE RAG PIPELINE (/v1/query)")
        print("==========================================================")
        
        query_payload = {
            "query": "What are Surya's primary technical skills and programming languages listed in the resume?",
            "tenant_id": "tenant_enterprise",
            "top_k": 10
        }
        
        print(f"Sending POST /v1/query: '{query_payload['query']}'...")
        q_resp = await client.post(
            "http://localhost:8000/v1/query",
            json=query_payload,
            headers={"X-Tenant-ID": "tenant_enterprise"}
        )
        print(f"Query Status Code: {q_resp.status_code}")
        try:
            res_json = q_resp.json()
            print("\n--- RETRIEVED ANSWER ---")
            print(res_json.get("answer"))
            print("\n--- CONFIDENCE & CITATIONS ---")
            print("Status:", res_json.get("confidence_status"))
            print("Citations:", json.dumps(res_json.get("citations"), indent=2))
        except Exception as e:
            print("Raw response:", q_resp.text)

if __name__ == "__main__":
    asyncio.run(main())
