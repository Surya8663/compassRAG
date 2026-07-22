"""
Corpus generator for the Evaluation Service.
Generates 5 realistic PDF documents using PyMuPDF (`fitz`) in `golden_corpus/`
covering directly answerable policies, OCR-dependent receipts, conflicting travel policies,
and ambiguous guidelines.
"""

from pathlib import Path

import fitz

CORPUS_DIR = Path(__file__).resolve().parent / "golden_corpus"


def generate_golden_corpus() -> list[Path]:
    """
    Creates all 5 evaluation PDF documents if they do not already exist.
    Returns list of absolute paths to the generated PDFs.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    # 1. handbook_2026.pdf (Directly Answerable)
    handbook_path = CORPUS_DIR / "handbook_2026.pdf"
    doc_handbook = fitz.open()
    # Page 1: Equipment Allowance
    p1 = doc_handbook.new_page()
    p1.insert_text(
        (50, 70),
        "Company IT Equipment Allowance Policy\n\n"
        "Every full-time employee is entitled to a $1,500 annual hardware stipend for\n"
        "purchasing laptops, monitors, and ergonomic accessories.\n"
        "All purchases must be submitted through the procurement portal.",
        fontsize=12,
    )
    # Page 2: Core Working Hours
    p2 = doc_handbook.new_page()
    p2.insert_text(
        (50, 70),
        "Core Working Hours Guidelines\n\n"
        "All staff must be available for synchronous collaboration between 10:00 AM\n"
        "and 3:00 PM EST Monday through Friday. Flexible working is permitted outside\n"
        "these core collaboration bands.",
        fontsize=12,
    )
    # Page 3: Annual Leave
    p3 = doc_handbook.new_page()
    p3.insert_text(
        (50, 70),
        "Annual Leave and Carry-Over Policy\n\n"
        "Employees receive 20 days of paid annual leave per year. Up to 5 unused\n"
        "leave days can be carried over to the following calendar year.\n"
        "Carry-over days must be utilized before March 31.",
        fontsize=12,
    )
    # Page 4: Wellness Stipend
    p4 = doc_handbook.new_page()
    p4.insert_text(
        (50, 70),
        "Employee Wellness Benefits\n\n"
        "The company provides a $75 monthly wellness stipend for gym memberships,\n"
        "mental health apps, and fitness tracking equipment.",
        fontsize=12,
    )
    doc_handbook.save(handbook_path)
    doc_handbook.close()
    generated_paths.append(handbook_path)

    # 2. reimbursement_receipt_scanned.pdf (OCR Dependent)
    receipt_path = CORPUS_DIR / "reimbursement_receipt_scanned.pdf"
    doc_receipt = fitz.open()
    p_receipt = doc_receipt.new_page()
    p_receipt.insert_text(
        (50, 70),
        "OFFICIAL REIMBURSEMENT INVOICE / RECEIPT\n\n"
        "VENDOR: Apex Catering Services Inc.\n"
        "TAX ID: 99-8877665\n"
        "DATE: 2026-03-15\n"
        "DESCRIPTION: Client Dinner and Team Lunch\n"
        "SUBTOTAL: $440.00\n"
        "TAX: $45.50\n"
        "TOTAL AMOUNT DUE: $485.50\n"
        "STATUS: PAID BY CORPORATE CREDIT CARD",
        fontsize=12,
    )
    doc_receipt.save(receipt_path)
    doc_receipt.close()
    generated_paths.append(receipt_path)

    # 3. travel_policy_v1_2024.pdf (Contradictory - Old Version)
    travel_v1_path = CORPUS_DIR / "travel_policy_v1_2024.pdf"
    doc_v1 = fitz.open()
    p_v1 = doc_v1.new_page()
    p_v1.insert_text(
        (50, 70),
        "Corporate Travel Guidelines (Version 1.0)\n"
        "Effective Date: January 1, 2024\n\n"
        "1. Hotel Accommodation: When traveling on official business, the maximum\n"
        "allowable hotel accommodation expense is $150 per night.\n"
        "2. Meal Per Diem: International meal per diem is capped at $65 per day.\n"
        "3. Flight Eligibility: Business class flights are only permitted for\n"
        "international flights exceeding 10 hours.",
        fontsize=12,
    )
    doc_v1.save(travel_v1_path)
    doc_v1.close()
    generated_paths.append(travel_v1_path)

    # 4. travel_policy_v2_2026.pdf (Contradictory - Newer Version)
    travel_v2_path = CORPUS_DIR / "travel_policy_v2_2026.pdf"
    doc_v2 = fitz.open()
    p_v2 = doc_v2.new_page()
    p_v2.insert_text(
        (50, 70),
        "Revised Corporate Travel Guidelines (Version 2.0)\n"
        "Effective Date: January 1, 2026\n"
        "Note: This policy supersedes Version 1.0 (2024).\n\n"
        "1. Hotel Accommodation: Due to increased hospitality costs, the maximum\n"
        "allowable hotel accommodation expense is now $250 per night.\n"
        "2. Meal Per Diem: International meal per diem is updated to $95 per day.\n"
        "3. Flight Eligibility: Business class flights are permitted for any\n"
        "international flight exceeding 8 hours.",
        fontsize=12,
    )
    doc_v2.save(travel_v2_path)
    doc_v2.close()
    generated_paths.append(travel_v2_path)

    # 5. remote_work_guidelines.pdf (Ambiguous)
    remote_path = CORPUS_DIR / "remote_work_guidelines.pdf"
    doc_remote = fitz.open()
    p_remote = doc_remote.new_page()
    p_remote.insert_text(
        (50, 70),
        "Remote Work Guidelines and Ad-Hoc Arrangements\n\n"
        "Employees may work remotely on ad-hoc days subject to manager discretion\n"
        "and operational requirements. Approval processes and notification deadlines\n"
        "vary by department, project phase, and team sprint schedule.",
        fontsize=12,
    )
    # 6. G_Surya_Resume.pdf (Surya Resume)
    resume_path = CORPUS_DIR / "G_Surya_Resume.pdf"
    doc_resume = fitz.open()
    p_resume = doc_resume.new_page()
    p_resume.insert_text(
        (50, 70),
        "G. Surya\n"
        "Bangalore, India | +91 86672 34480 | iamsurya195@gmail.com\n\n"
        "Summary\n"
        "Generative AI Engineer building RAG systems and agentic workflows.\n\n"
        "Experience\n"
        "Generative AI Developer Intern Nov 2025 – May 2026\n"
        "HiDevs Bangalore, India (Hybrid)\n"
        "– Built PeopleGPT / Aura AI Agent – RAG pipeline over 13,000+ profiles using Qdrant hybrid search, LangChain, FastAPI (SSE); cut manual search time by 70%.\n"
        "– Optimized Dave Chatbot latency from 10–15s to ~2.5s via code review; added streaming and MongoDB session persistence.\n"
        "– Built AI GitHub Repository Analyzer (AST checks, commit analysis, Gemini summaries); reduced codebase review time by 40%.\n",
        fontsize=11,
    )
    doc_resume.save(resume_path)
    doc_resume.close()
    generated_paths.append(resume_path)

    return generated_paths


if __name__ == "__main__":
    paths = generate_golden_corpus()
    print(f"Generated {len(paths)} PDF documents in {CORPUS_DIR}")
