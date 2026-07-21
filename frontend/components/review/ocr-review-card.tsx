"use client";

import React, { useState } from "react";
import { type OcrReviewItem } from "@/lib/types";
import { Check, Trash2, AlertTriangle, Eye, RefreshCw, ZoomIn, FileText, Cpu } from "lucide-react";
import { toast } from "sonner";

interface OcrReviewCardProps {
  item: OcrReviewItem;
  onApprove: (id: string, correctedText: string) => void;
  onDiscard: (id: string) => void;
}

export function OcrReviewCard({ item, onApprove, onDiscard }: OcrReviewCardProps) {
  const [text, setText] = useState(item.extracted_text);
  const [isZoomed, setIsZoomed] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleApprove = () => {
    setIsSubmitting(true);
    const toastId = toast.loading("Submitting corrected OCR text to Qdrant indexer...");
    setTimeout(() => {
      onApprove(item.page_id, text);
      toast.success("Page approved & re-indexed successfully!", {
        id: toastId,
        description: `Updated embeddings in Qdrant and BM25 index for ${item.document_name} (Page ${item.page_number})`,
      });
      setIsSubmitting(false);
    }, 600);
  };

  const handleDiscard = () => {
    onDiscard(item.page_id);
    toast.success("Flagged page discarded from active index");
  };

  return (
    <div className="rounded-2xl border border-border/80 bg-card/90 shadow-lg backdrop-blur-xl overflow-hidden transition-all">
      {/* Card Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 bg-background/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-orange-500/10 text-orange-500 border border-orange-500/20">
            <AlertTriangle className="h-5 w-5" />
          </div>
          <div>
            <h4 className="font-display text-base font-bold text-foreground">
              {item.document_name} — Page {item.page_number}
            </h4>
            <p className="text-xs text-muted-foreground flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-primary" />
              <span>Tesseract Confidence: </span>
              <span className="font-mono font-bold text-orange-500">
                {(item.confidence_score * 100).toFixed(0)}%
              </span>
              <span>(Threshold &lt; 85%)</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDiscard}
            disabled={isSubmitting}
            className="flex items-center gap-1.5 rounded-xl border border-border/80 bg-card px-3.5 py-2 text-xs font-semibold text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Discard Page</span>
          </button>

          <button
            onClick={handleApprove}
            disabled={isSubmitting}
            className="flex items-center gap-1.5 rounded-xl bg-primary px-4 py-2 text-xs font-bold text-primary-foreground hover:bg-primary/90 shadow-md shadow-primary/25 transition-all"
          >
            {isSubmitting ? (
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Check className="h-3.5 w-3.5" />
            )}
            <span>Approve & Re-index</span>
          </button>
        </div>
      </div>

      {/* Flagged Reason Alert */}
      <div className="border-b border-border/40 bg-orange-500/5 px-6 py-2.5 text-xs text-orange-500/90 flex items-center gap-2">
        <span className="font-bold uppercase tracking-wider text-[10px] rounded bg-orange-500/20 px-1.5 py-0.5 border border-orange-500/30">
          Flagged Reason
        </span>
        <span>{item.flagged_reason}</span>
      </div>

      {/* Side-by-Side Review Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-border/60">
        {/* Left: Scanned Image Preview */}
        <div className="flex flex-col bg-background/40 p-6">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Eye className="h-3.5 w-3.5 text-primary" /> Scanned Document Image Preview
            </span>
            <button
              onClick={() => setIsZoomed(!isZoomed)}
              className="flex items-center gap-1 rounded-lg border border-border/60 bg-card px-2.5 py-1 text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              <ZoomIn className="h-3 w-3" />
              <span>{isZoomed ? "Zoom Out" : "Zoom In"}</span>
            </button>
          </div>

          <div className="relative flex-1 rounded-xl border border-border/60 bg-black/60 overflow-hidden min-h-[340px] flex items-center justify-center">
            <img
              src={item.scanned_image_url}
              alt={`Scanned page ${item.page_number}`}
              className={`max-h-[380px] w-auto object-contain transition-transform duration-300 ${
                isZoomed ? "scale-150 cursor-zoom-out" : "scale-100 cursor-zoom-in"
              }`}
              onClick={() => setIsZoomed(!isZoomed)}
            />
          </div>
        </div>

        {/* Right: Editable OCR Text Area */}
        <div className="flex flex-col bg-card/40 p-6 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5 text-primary" /> Editable Extracted OCR Text
            </span>
            <span className="text-[11px] font-mono text-muted-foreground/80">
              {text.length} characters
            </span>
          </div>

          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={14}
            disabled={isSubmitting}
            className="w-full flex-1 rounded-xl border border-border/80 bg-background/80 p-4 font-mono text-xs text-foreground focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary shadow-inner leading-relaxed resize-none"
            placeholder="Correct OCR text errors here before approving..."
          />
        </div>
      </div>
    </div>
  );
}
