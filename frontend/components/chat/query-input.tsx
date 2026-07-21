"use client";

import React, { useState } from "react";
import { Send, Sparkles, SlidersHorizontal, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

interface QueryInputProps {
  onSend: (question: string) => void;
  isLoading: boolean;
}

const QUICK_PROMPTS = [
  { label: "Contradiction Check", prompt: "What is the remote work equipment reimbursement cap for new hires?" },
  { label: "Ambiguity Clarification", prompt: "Does the annual bonus calculation include overtime differentials?" },
  { label: "Verified Citation", prompt: "What multi-factor authentication (MFA) standards are mandated for production?" },
  { label: "Low Confidence / Abstention", prompt: "What is the exact salary of the Chief Financial Officer for FY2027?" },
];

export function QueryInput({ onSend, isLoading }: QueryInputProps) {
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(5);
  const [showOptions, setShowOptions] = useState(false);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="space-y-3">
      {/* Quick Action Prompt Chips */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar">
        <span className="text-[11px] font-semibold uppercase text-muted-foreground shrink-0 flex items-center gap-1">
          <Sparkles className="h-3 w-3 text-primary" /> Test Scenarios:
        </span>
        {QUICK_PROMPTS.map((qp, idx) => (
          <button
            key={idx}
            type="button"
            disabled={isLoading}
            onClick={() => {
              setInput(qp.prompt);
              onSend(qp.prompt);
              setInput("");
            }}
            className="shrink-0 rounded-full border border-border/60 bg-card/80 px-3 py-1 text-xs font-medium text-muted-foreground hover:border-primary/50 hover:bg-primary/10 hover:text-primary transition-all shadow-sm"
          >
            {qp.label}
          </button>
        ))}
      </div>

      {/* Main Input Box */}
      <form onSubmit={handleSubmit} className="relative rounded-2xl border border-border/80 bg-card/90 shadow-lg focus-within:border-primary/80 focus-within:ring-2 focus-within:ring-primary/20 transition-all backdrop-blur-xl">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything across enterprise documents... (e.g. 'What is the data retention schedule for WORM snapshots?')"
          rows={3}
          disabled={isLoading}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-12 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
        />

        {/* Bottom Toolbar inside input */}
        <div className="absolute bottom-2.5 left-3 right-3 flex items-center justify-between border-t border-border/40 pt-2">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowOptions(!showOptions)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors border border-border/60",
                showOptions ? "bg-primary text-primary-foreground" : "bg-background/50 text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span>Top-K: {topK}</span>
            </button>

            <span className="hidden sm:flex items-center gap-1 text-[10px] text-muted-foreground/80">
              <Cpu className="h-3 w-3 text-emerald-500" /> Hybrid RRF + Cohere Rerank + NLI Guard
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden md:inline-block text-[11px] text-muted-foreground/70 font-mono">
              Press <kbd className="rounded border border-border/80 bg-background/80 px-1 py-0.5 text-[10px] font-sans">Enter ↵</kbd> to submit
            </span>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={cn(
                "flex h-8 items-center gap-1.5 rounded-xl px-3.5 text-xs font-semibold transition-all shadow-md",
                !input.trim() || isLoading
                  ? "bg-muted text-muted-foreground cursor-not-allowed opacity-60"
                  : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-primary/25"
              )}
            >
              <span>Verify & Query</span>
              <Send className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </form>

      {/* Advanced Strategy Options Drawer */}
      {showOptions && (
        <div className="rounded-xl border border-border/60 bg-card/80 p-3.5 shadow-sm space-y-3 animate-accordion-down text-xs">
          <div className="flex items-center justify-between font-semibold text-foreground">
            <span>Retrieval & Correction Configuration</span>
            <span className="text-[10px] uppercase font-mono text-primary">LangGraph StateGraph Engine</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <label className="text-muted-foreground font-medium">Candidate Top-K</label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full rounded-lg border border-border/60 bg-background px-2 py-1 text-foreground focus:outline-none focus:border-primary"
              >
                <option value={3}>3 Chunks (Fastest)</option>
                <option value={5}>5 Chunks (Balanced)</option>
                <option value={10}>10 Chunks (Deep Scan)</option>
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-muted-foreground font-medium">Contradiction NLI Model</label>
              <div className="rounded-lg border border-border/60 bg-background/50 px-2.5 py-1 text-foreground font-mono">
                GPT-4o-mini structured NLI
              </div>
            </div>
            <div className="space-y-1">
              <label className="text-muted-foreground font-medium">Groundedness Threshold</label>
              <div className="rounded-lg border border-border/60 bg-background/50 px-2.5 py-1 text-emerald-500 font-semibold">
                score &gt;= 0.80 (Atomic Claims)
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
