"use client";

import React, { useState, useEffect } from "react";
import { OcrReviewCard } from "@/components/review/ocr-review-card";
import { type OcrReviewItem } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { CheckCircle2, ShieldCheck } from "lucide-react";

export default function ReviewPage() {
  const [items, setItems] = useState<OcrReviewItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const queue = apiClient.getReviewQueue();
    setItems(queue);
    setIsLoading(false);
  }, []);

  const handleApprove = (id: string, correctedText: string) => {
    console.log("Approved OCR text with characters:", correctedText.length);
    setItems((prev) => prev.filter((i) => i.page_id !== id));
  };

  const handleDiscard = (id: string) => {
    setItems((prev) => prev.filter((i) => i.page_id !== id));
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Top Banner / Executive Overview */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-2xl border border-border/60 bg-gradient-to-r from-card via-card/90 to-orange-500/5 p-6 shadow-sm backdrop-blur-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-orange-500/20 px-2 py-0.5 text-[10px] font-bold text-orange-500 border border-orange-500/30 uppercase tracking-wider">
              Human-In-The-Loop
            </span>
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-500" /> Admin Oversight
            </span>
          </div>
          <h2 className="font-display text-2xl font-bold tracking-tight text-foreground">
            Manual OCR Correction Queue
          </h2>
          <p className="text-xs text-muted-foreground max-w-2xl">
            When our Tesseract OCR confidence falls below the strict 85% threshold, scanned pages are flagged right here for human review before vectorization. Correct extraction discrepancies and submit directly to Qdrant.
          </p>
        </div>

        <div className="flex items-center gap-3 self-stretch md:self-auto justify-end">
          <div className="rounded-xl border border-border/60 bg-card/80 px-4 py-2.5 text-center shadow-sm">
            <div className="text-lg font-bold font-display text-orange-500">
              {items.length}
            </div>
            <div className="text-[11px] font-medium text-muted-foreground">Pending Review</div>
          </div>
        </div>
      </div>

      {/* Review Cards Feed */}
      {isLoading ? (
        <div className="space-y-6">
          {[1, 2].map((i) => (
            <div key={i} className="h-96 rounded-2xl border border-border/60 bg-card/50 animate-pulse" />
          ))}
        </div>
      ) : items.length === 0 ? (
        /* Designed Empty Queue State */
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/80 bg-card/40 p-16 text-center shadow-sm">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20 mb-4 shadow-sm">
            <CheckCircle2 className="h-8 w-8" />
          </div>
          <h3 className="font-display text-lg font-bold text-foreground">
            OCR Review Queue is Clean
          </h3>
          <p className="mt-1 max-w-md text-xs text-muted-foreground">
            All scanned document pages have achieved &gt;85% confidence or have been manually verified and indexed into Qdrant. No further human intervention required.
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {items.map((item) => (
            <OcrReviewCard
              key={item.page_id}
              item={item}
              onApprove={handleApprove}
              onDiscard={handleDiscard}
            />
          ))}
        </div>
      )}
    </div>
  );
}
