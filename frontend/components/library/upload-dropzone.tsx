"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, Loader2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { type IngestedDocument } from "@/lib/types";
import { apiClient } from "@/lib/api";

interface UploadDropzoneProps {
  onDocumentIngested: (doc: IngestedDocument) => void;
}

export function UploadDropzone({ onDocumentIngested }: UploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await validateAndUpload(file);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await validateAndUpload(file);
    }
  };

  const validateAndUpload = async (file: File) => {
    if (!file.name.endsWith(".pdf") && !file.name.endsWith(".txt") && !file.name.endsWith(".md")) {
      toast.error("Unsupported file type", {
        description: "Please upload PDF, TXT, or Markdown documents.",
      });
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      toast.error("File exceeds maximum size limit (50MB)");
      return;
    }

    setSelectedFile(file);
    setIsUploading(true);

    const toastId = toast.loading(`Uploading "${file.name}" to ingestion pipeline...`);

    try {
      const result = await apiClient.ingestDocument(file);
      toast.success("Ingestion job started successfully!", {
        id: toastId,
        description: `Job ID: ${result.job_id} — Chunking & embedding via Qdrant/ES`,
      });

      const newDoc: IngestedDocument = {
        id: result.document_id || `doc_${Math.random().toString(36).substring(2, 8)}`,
        job_id: result.job_id,
        name: file.name,
        size_bytes: file.size,
        status: "PROCESSING",
        progress: 15,
        chunks_indexed: 0,
        created_at: "Just now",
      };

      onDocumentIngested(newDoc);
      setSelectedFile(null);
    } catch (err) {
      toast.error("Failed to upload document", {
        id: toastId,
        description: err instanceof Error ? err.message : "Unknown backend error",
      });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !isUploading && fileInputRef.current?.click()}
      className={cn(
        "group relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 transition-all duration-300 cursor-pointer text-center overflow-hidden",
        isDragging
          ? "border-primary bg-primary/10 scale-[1.01] shadow-lg shadow-primary/20"
          : "border-border/80 bg-card/60 hover:border-primary/50 hover:bg-card/90 shadow-sm"
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Background Subtle Gradient Glow */}
      <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-transparent to-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />

      {/* Icon Graphic */}
      <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-indigo-500/10 text-primary mb-4 shadow-inner group-hover:scale-110 transition-transform duration-300">
        {isUploading ? (
          <Loader2 className="h-7 w-7 animate-spin text-primary" />
        ) : (
          <UploadCloud className="h-7 w-7" />
        )}
        {!isUploading && (
          <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[9px] font-bold text-primary-foreground shadow">
            +
          </span>
        )}
      </div>

      <div className="space-y-1 z-10 max-w-md">
        <h3 className="font-display text-base font-semibold text-foreground">
          {isUploading ? "Processing Document Ingestion..." : "Drag & drop enterprise documents here"}
        </h3>
        <p className="text-xs text-muted-foreground">
          {isUploading ? (
            `Parsing layout and extracting semantic vectors for ${selectedFile?.name}...`
          ) : (
            <span>
              Supports <strong className="text-foreground">PDF</strong>, <strong className="text-foreground">TXT</strong>, and <strong className="text-foreground">Markdown</strong> (up to 50MB). Automated OCR fallback enabled for scanned pages.
            </span>
          )}
        </p>
      </div>

      {!isUploading && (
        <div className="mt-5 flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground shadow-md shadow-primary/25 group-hover:bg-primary/90 transition-all">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Browse Knowledgebase Files</span>
        </div>
      )}
    </div>
  );
}
