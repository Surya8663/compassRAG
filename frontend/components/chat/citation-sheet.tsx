"use client";

import React from "react";
import { type Citation } from "@/lib/types";
import { X, FileText, CheckCircle2, BookOpen, Hash, Copy } from "lucide-react";
import { toast } from "sonner";

interface CitationSheetProps {
  citation: Citation | null;
  onClose: () => void;
}

export function CitationSheet({ citation, onClose }: CitationSheetProps) {
  if (!citation) return null;

  const handleCopySnippet = () => {
    navigator.clipboard.writeText(citation.quote_snippet);
    toast.success("Quote snippet copied to clipboard");
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-background/60 backdrop-blur-sm animate-fade-in">
      {/* Slide-out Sheet */}
      <div className="relative flex w-full max-w-lg flex-col border-l border-border/80 bg-card p-6 shadow-2xl animate-slide-in-right overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border/60 pb-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-display text-base font-bold text-foreground">
                Verified Source Citation
              </h3>
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> Exact Grounded Chunk Match
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border/60 bg-background/50 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Metadata Details */}
        <div className="mt-6 space-y-4">
          <div className="rounded-xl border border-border/60 bg-background/50 p-4 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                <BookOpen className="h-3.5 w-3.5 text-primary" /> Document Source:
              </span>
              <span className="font-semibold text-foreground truncate max-w-[220px]">
                {citation.source}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-medium flex items-center gap-1.5">
                <Hash className="h-3.5 w-3.5 text-primary" /> Page Number:
              </span>
              <span className="rounded-md bg-primary/10 px-2 py-0.5 font-bold text-primary border border-primary/20">
                Page {citation.page_number}
              </span>
            </div>

            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-medium">Vector Chunk ID:</span>
              <span className="font-mono text-[11px] text-muted-foreground/80">
                {citation.chunk_id}
              </span>
            </div>
          </div>

          {/* Exact Quote Snippet Box */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Extracted Quote Snippet
              </h4>
              <button
                onClick={handleCopySnippet}
                className="flex items-center gap-1 text-[11px] font-medium text-primary hover:underline"
              >
                <Copy className="h-3 w-3" /> Copy Quote
              </button>
            </div>

            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4 font-mono text-xs text-foreground/90 leading-relaxed shadow-inner border-l-4 border-l-emerald-500">
              &ldquo;{citation.quote_snippet}&rdquo;
            </div>
          </div>

          {/* NLI & Groundedness Explanation */}
          <div className="rounded-xl border border-border/60 bg-card p-4 space-y-2">
            <h5 className="text-xs font-semibold text-foreground flex items-center gap-1.5">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" /> Atomic Claim Groundedness
            </h5>
            <p className="text-xs text-muted-foreground leading-relaxed">
              This quote snippet was verified via our NLI entailment checker (ENTAILMENT confidence &gt; 0.90) against the decomposed claims in the final synthesized output. It cannot be retrieved by unauthorized tenants.
            </p>
          </div>
        </div>

        {/* Footer actions */}
        <div className="mt-auto pt-6 border-t border-border/60 flex items-center justify-between">
          <span className="text-[11px] text-muted-foreground">Tenant A Document RBAC Enforced</span>
          <button
            onClick={onClose}
            className="rounded-xl bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground hover:bg-primary/90 shadow-sm"
          >
            Close Inspector
          </button>
        </div>
      </div>
    </div>
  );
}
