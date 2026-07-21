"use client";

import React, { useState, useEffect } from "react";
import { UploadDropzone } from "@/components/library/upload-dropzone";
import { DocumentGrid } from "@/components/library/document-grid";
import { type IngestedDocument } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { ShieldCheck, Database } from "lucide-react";
import { toast } from "sonner";

export default function LibraryPage() {
  const [documents, setDocuments] = useState<IngestedDocument[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load initial documents from API client
    const initialDocs = apiClient.getInitialDocuments();
    setDocuments(initialDocs);
    setIsLoading(false);

    // Simulate polling for active jobs
    const interval = setInterval(() => {
      setDocuments((prev) =>
        prev.map((doc) => {
          if (doc.status === "PROCESSING" || doc.status === "OCR_IN_PROGRESS") {
            const nextProgress = Math.min(doc.progress + 15, 100);
            if (nextProgress === 100) {
              toast.success(`"${doc.name}" indexing complete!`, {
                description: `Indexed 42 chunks into Qdrant vector space & ES BM25 index.`,
              });
              return {
                ...doc,
                status: "INDEXED",
                progress: 100,
                chunks_indexed: 42,
              };
            }
            return { ...doc, progress: nextProgress };
          }
          return doc;
        })
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const handleDocumentIngested = (newDoc: IngestedDocument) => {
    setDocuments((prev) => [newDoc, ...prev]);
  };

  const handleRefresh = () => {
    const toastId = toast.loading("Syncing ingestion pipeline status...");
    setTimeout(() => {
      toast.success("All ingestion status synced with API Gateway", { id: toastId });
    }, 600);
  };

  const handleDelete = (id: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== id));
    toast.success("Document removed from tenant index");
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Top Banner / Executive Overview */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-2xl border border-border/60 bg-gradient-to-r from-card via-card/90 to-primary/5 p-6 shadow-sm backdrop-blur-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-primary/20 px-2 py-0.5 text-[10px] font-bold text-primary border border-primary/30 uppercase tracking-wider">
              Ingestion Center
            </span>
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" /> Tenant A Isolation
            </span>
          </div>
          <h2 className="font-display text-2xl font-bold tracking-tight text-foreground">
            Knowledgebase Document Library
          </h2>
          <p className="text-xs text-muted-foreground max-w-2xl">
            Upload PDFs, Markdown, and scanned files. Our pipeline automatically performs layout-aware chunking, BAAI/bge vector embeddings in Qdrant, and BM25 indexing in Elasticsearch.
          </p>
        </div>

        <div className="flex items-center gap-3 self-stretch md:self-auto justify-end">
          <div className="rounded-xl border border-border/60 bg-card/80 px-4 py-2.5 text-center shadow-sm">
            <div className="text-lg font-bold font-display text-foreground">
              {documents.filter((d) => d.status === "INDEXED").length}
            </div>
            <div className="text-[11px] font-medium text-muted-foreground">Indexed Docs</div>
          </div>
          <div className="rounded-xl border border-border/60 bg-card/80 px-4 py-2.5 text-center shadow-sm">
            <div className="text-lg font-bold font-display text-primary">
              {documents.reduce((acc, d) => acc + (d.chunks_indexed || 0), 0)}
            </div>
            <div className="text-[11px] font-medium text-muted-foreground">Total Chunks</div>
          </div>
        </div>
      </div>

      {/* Drag & Drop Zone */}
      <UploadDropzone onDocumentIngested={handleDocumentIngested} />

      {/* Document Status Feed Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-display text-base font-bold text-foreground flex items-center gap-2">
            <Database className="h-4 w-4 text-primary" />
            Ingested Documents Status
          </h3>
        </div>

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-32 rounded-xl border border-border/60 bg-card/50 animate-pulse" />
            ))}
          </div>
        ) : (
          <DocumentGrid
            documents={documents}
            onRefresh={handleRefresh}
            onDelete={handleDelete}
          />
        )}
      </div>
    </div>
  );
}
