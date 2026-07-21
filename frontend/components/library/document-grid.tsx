"use client";

import React, { useState } from "react";
import { 
  FileText, 
  CheckCircle2, 
  AlertTriangle, 
  RefreshCw, 
  Trash2, 
  Search, 
  Database,
  ArrowUpRight,
  Cpu
} from "lucide-react";
import { type IngestedDocument } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DocumentGridProps {
  documents: IngestedDocument[];
  onRefresh: () => void;
  onDelete?: (id: string) => void;
}

export function DocumentGrid({ documents, onRefresh, onDelete }: DocumentGridProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");

  const filteredDocs = documents.filter((doc) => {
    const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase());
    if (statusFilter === "ALL") return matchesSearch;
    if (statusFilter === "INDEXED") return matchesSearch && doc.status === "INDEXED";
    if (statusFilter === "PROCESSING") return matchesSearch && (doc.status === "PROCESSING" || doc.status === "OCR_IN_PROGRESS");
    if (statusFilter === "REVIEW") return matchesSearch && doc.status === "NEEDS_REVIEW";
    return matchesSearch;
  });

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const getStatusBadge = (doc: IngestedDocument) => {
    switch (doc.status) {
      case "INDEXED":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-500">
            <CheckCircle2 className="h-3 w-3" />
            Indexed ({doc.chunks_indexed || 0} chunks)
          </span>
        );
      case "PROCESSING":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-2.5 py-0.5 text-[11px] font-semibold text-primary">
            <RefreshCw className="h-3 w-3 animate-spin" />
            Processing ({doc.progress}%)
          </span>
        );
      case "OCR_IN_PROGRESS":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-amber-500">
            <Cpu className="h-3 w-3 animate-pulse" />
            Tesseract OCR ({doc.progress}%)
          </span>
        );
      case "NEEDS_REVIEW":
        return (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-orange-500/30 bg-orange-500/10 px-2.5 py-0.5 text-[11px] font-semibold text-orange-500">
            <AlertTriangle className="h-3 w-3 animate-bounce" />
            Needs OCR Review
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 rounded-full border border-border px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground">
            {doc.status}
          </span>
        );
    }
  };

  return (
    <div className="space-y-5">
      {/* Filter Tabs & Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 border-b border-border/60 pb-4">
        {/* Tabs */}
        <div className="flex items-center gap-1.5 rounded-xl border border-border/60 bg-card p-1 text-xs font-semibold shadow-sm overflow-x-auto">
          {[
            { id: "ALL", label: "All Documents", count: documents.length },
            { id: "INDEXED", label: "Indexed", count: documents.filter((d) => d.status === "INDEXED").length },
            { id: "PROCESSING", label: "Processing / OCR", count: documents.filter((d) => d.status === "PROCESSING" || d.status === "OCR_IN_PROGRESS").length },
            { id: "REVIEW", label: "Needs Review", count: documents.filter((d) => d.status === "NEEDS_REVIEW").length },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setStatusFilter(tab.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-colors whitespace-nowrap",
                statusFilter === tab.id
                  ? "bg-primary text-primary-foreground shadow"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <span>{tab.label}</span>
              <span
                className={cn(
                  "rounded-full px-1.5 py-0.2 text-[10px]",
                  statusFilter === tab.id
                    ? "bg-primary-foreground/20 text-primary-foreground"
                    : "bg-muted text-muted-foreground"
                )}
              >
                {tab.count}
              </span>
            </button>
          ))}
        </div>

        {/* Search & Refresh */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1 sm:w-64">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search knowledgebase..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-xl border border-border/60 bg-card py-2 pl-9 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-sm"
            />
          </div>
          <button
            onClick={onRefresh}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-border/60 bg-card text-muted-foreground hover:bg-accent hover:text-foreground transition-colors shadow-sm"
            title="Refresh ingestion status"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Document Grid / Table View */}
      {filteredDocs.length === 0 ? (
        /* Designed Empty State */
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/80 bg-card/40 p-12 text-center shadow-sm">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent text-muted-foreground mb-4">
            <Database className="h-7 w-7" />
          </div>
          <h4 className="font-display text-base font-semibold text-foreground">
            No documents found in target filter
          </h4>
          <p className="mt-1 max-w-sm text-xs text-muted-foreground">
            Try adjusting your search query or switch tab filters above. You can also drag and drop new documents above to begin indexing.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
          {filteredDocs.map((doc) => (
            <div
              key={doc.id}
              className="group relative flex flex-col justify-between rounded-xl border border-border/60 bg-card/90 p-5 shadow-sm hover:border-primary/40 hover:shadow-md transition-all duration-200"
            >
              <div>
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
                      <FileText className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <h4 className="font-semibold text-sm text-foreground truncate group-hover:text-primary transition-colors">
                        {doc.name}
                      </h4>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                        <span>{formatBytes(doc.size_bytes)}</span>
                        <span>•</span>
                        <span>{doc.created_at}</span>
                        <span>•</span>
                        <span className="font-mono text-[10px] uppercase text-muted-foreground/80">
                          {doc.job_id}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    {onDelete && (
                      <button
                        onClick={() => onDelete(doc.id)}
                        className="opacity-0 group-hover:opacity-100 flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-all"
                        title="Delete document"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Progress Bar (if processing or OCR) */}
                {(doc.status === "PROCESSING" || doc.status === "OCR_IN_PROGRESS") && (
                  <div className="mt-4 space-y-1.5">
                    <div className="flex justify-between text-[11px] font-medium text-muted-foreground">
                      <span>Pipeline Progress</span>
                      <span className="text-primary font-semibold">{doc.progress}%</span>
                    </div>
                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                      <div
                        className="h-full bg-gradient-to-r from-primary to-indigo-500 transition-all duration-500 rounded-full"
                        style={{ width: `${doc.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-4 flex items-center justify-between border-t border-border/40 pt-3">
                {getStatusBadge(doc)}

                {doc.status === "NEEDS_REVIEW" && (
                  <a
                    href="/review"
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-orange-500 hover:underline"
                  >
                    <span>Open Review Queue</span>
                    <ArrowUpRight className="h-3 w-3" />
                  </a>
                )}
                {doc.status === "INDEXED" && (
                  <span className="text-[11px] font-mono text-muted-foreground/80 flex items-center gap-1">
                    <Database className="h-3 w-3 text-emerald-500" /> Tenant A Isolated
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
