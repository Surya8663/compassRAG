"use client";

import React, { useState } from "react";
import { type Citation, type ConfidenceStatus, type QueryResponse } from "@/lib/types";
import { motion } from "framer-motion";
import { 
  CheckCircle2, 
  HelpCircle, 
  AlertTriangle, 
  Sparkles, 
  Clock, 
  Cpu, 
  Send, 
  FileText,
  CornerDownRight
} from "lucide-react";
import { cn, formatDuration } from "@/lib/utils";

interface ResponseCardProps {
  question: string;
  response: QueryResponse;
  onCitationClick: (citation: Citation) => void;
  onFollowupReply?: (replyText: string) => void;
}

export function ResponseCard({
  question,
  response,
  onCitationClick,
  onFollowupReply,
}: ResponseCardProps) {
  const [replyInput, setReplyInput] = useState("");

  const handleReplySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyInput.trim() || !onFollowupReply) return;
    onFollowupReply(replyInput.trim());
    setReplyInput("");
  };

  // Render markdown-like text with clickable inline superscript citation badges [1], [2]
  const renderFormattedAnswer = (text: string, citations: Citation[]) => {
    // Replace citation tags like [1] or [2] with interactive buttons
    const parts = text.split(/(\[\d+\])/g);

    return parts.map((part, idx) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citIndex = parseInt(match[1], 10) - 1;
        const cit = citations[citIndex];
        if (cit) {
          return (
            <button
              key={idx}
              onClick={() => onCitationClick(cit)}
              className="inline-flex items-center justify-center rounded-md border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 mx-0.5 text-[11px] font-bold text-emerald-500 hover:bg-emerald-500 hover:text-white transition-all shadow-sm align-baseline cursor-pointer"
              title={`Click to inspect quote: "${cit.source} (Page ${cit.page_number})"`}
            >
              [{match[1]}]
            </button>
          );
        }
      }

      // Basic bold formatting support (**bold**)
      const boldParts = part.split(/(\*\*.*?\*\*)/g);
      return (
        <span key={idx}>
          {boldParts.map((bp, bIdx) => {
            if (bp.startsWith("**") && bp.endsWith("**")) {
              return <strong key={bIdx} className="font-semibold text-foreground">{bp.slice(2, -2)}</strong>;
            }
            return bp;
          })}
        </span>
      );
    });
  };

  const getCardStyle = (status: ConfidenceStatus) => {
    switch (status) {
      case "VERIFIED":
        return {
          border: "border-emerald-500/35 hover:border-emerald-500/50",
          bg: "bg-gradient-to-br from-card via-card/95 to-emerald-500/5",
          badgeBg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-500",
          badgeIcon: CheckCircle2,
          badgeLabel: "VERIFIED & GROUNDED",
          accentColor: "text-emerald-500",
        };
      case "CLARIFICATION_NEEDED":
        return {
          border: "border-amber-500/40 hover:border-amber-500/60",
          bg: "bg-gradient-to-br from-card via-card/95 to-amber-500/5",
          badgeBg: "bg-amber-500/10 border-amber-500/30 text-amber-500",
          badgeIcon: HelpCircle,
          badgeLabel: "CLARIFICATION NEEDED",
          accentColor: "text-amber-500",
        };
      case "LOW_CONFIDENCE":
        return {
          border: "border-rose-500/35 hover:border-rose-500/50",
          bg: "bg-gradient-to-br from-card via-card/95 to-rose-500/5",
          badgeBg: "bg-rose-500/10 border-rose-500/30 text-rose-500",
          badgeIcon: AlertTriangle,
          badgeLabel: "LOW CONFIDENCE / ABSTAINED",
          accentColor: "text-rose-500",
        };
    }
  };

  const style = getCardStyle(response.confidence_status);
  const BadgeIcon = style.badgeIcon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="space-y-4"
    >
      {/* User Question Bubble (Right Aligned) */}
      <div className="flex justify-end">
        <div className="max-w-xl rounded-2xl rounded-tr-sm bg-primary px-5 py-3.5 text-sm font-medium text-primary-foreground shadow-md shadow-primary/20">
          {question}
        </div>
      </div>

      {/* AI Semantic Response Card (Left Aligned) */}
      <div className={cn("relative rounded-2xl border p-6 shadow-lg backdrop-blur-xl transition-all duration-300", style.border, style.bg)}>
        {/* Top Header Row with Status Badge & Latency */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/40 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary border border-primary/20">
              <Sparkles className="h-4 w-4" />
            </div>
            <span className="font-display text-sm font-bold text-foreground">
              Compass AI Assistant
            </span>

            {/* Semantic Status Badge */}
            <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-bold tracking-wide shadow-sm", style.badgeBg)}>
              <BadgeIcon className="h-3.5 w-3.5" />
              <span>{style.badgeLabel}</span>
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1 font-mono">
              <Clock className="h-3.5 w-3.5 text-primary" />
              {formatDuration(response.latency_ms)}
            </span>
            <span>•</span>
            <span className="font-mono text-[11px] uppercase">
              Score: {
                response.confidence_score !== undefined && response.confidence_score !== null && !isNaN(response.confidence_score) && isFinite(response.confidence_score)
                  ? `${Math.min(100, Math.max(0, Math.round(response.confidence_score * 100)))}%`
                  : "Unavailable"
              }
            </span>
          </div>
        </div>

        {/* Answer Content Area */}
        <div className="mt-4 text-sm leading-relaxed text-foreground/95 whitespace-pre-line">
          {renderFormattedAnswer(response.answer, response.citations)}
        </div>

        {/* Reasoning / Contradiction Reconciliation Note */}
        {response.reasoning && (
          <div className="mt-5 rounded-xl border border-border/60 bg-background/60 p-3.5 text-xs text-muted-foreground flex items-start gap-2.5 shadow-sm">
            <Cpu className="h-4 w-4 shrink-0 text-primary mt-0.5" />
            <div>
              <span className="font-semibold text-foreground">Self-Correction Audit Note: </span>
              <span>{response.reasoning}</span>
            </div>
          </div>
        )}

        {/* Citations Row (If present) */}
        {response.citations && response.citations.length > 0 && (
          <div className="mt-5 border-t border-border/40 pt-4 space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5 text-primary" />
              <span>Verified Grounded Citations (Click to Inspect)</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {response.citations.map((cit, idx) => (
                <button
                  key={idx}
                  onClick={() => onCitationClick(cit)}
                  className="group flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-500 hover:bg-emerald-500 hover:text-white transition-all shadow-sm"
                >
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 text-[10px] font-bold group-hover:bg-white/20">
                    {idx + 1}
                  </span>
                  <span className="truncate max-w-[180px]">{cit.source}</span>
                  <span className="font-mono opacity-80">(P. {cit.page_number})</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Clarification Reply Affordance (when CLARIFICATION_NEEDED) */}
        {response.confidence_status === "CLARIFICATION_NEEDED" && onFollowupReply && (
          <div className="mt-6 border-t border-amber-500/30 pt-4">
            <form onSubmit={handleReplySubmit} className="flex items-center gap-2">
              <div className="relative flex-1">
                <CornerDownRight className="absolute left-3 top-3 h-4 w-4 text-amber-500" />
                <input
                  type="text"
                  value={replyInput}
                  onChange={(e) => setReplyInput(e.target.value)}
                  placeholder="Type your clarification (e.g., 'Exempt salaried personnel')..."
                  className="w-full rounded-xl border border-amber-500/40 bg-background/80 py-2.5 pl-9 pr-4 text-xs text-foreground placeholder:text-muted-foreground focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 shadow-inner"
                />
              </div>
              <button
                type="submit"
                disabled={!replyInput.trim()}
                className="flex h-10 items-center gap-1.5 rounded-xl bg-amber-500 px-4 text-xs font-bold text-amber-950 hover:bg-amber-400 disabled:opacity-50 transition-all shadow-sm"
              >
                <span>Reply</span>
                <Send className="h-3.5 w-3.5" />
              </button>
            </form>
          </div>
        )}
      </div>
    </motion.div>
  );
}
