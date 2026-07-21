"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Cpu, ShieldCheck, RefreshCw, Layers } from "lucide-react";

interface ThinkingStatusProps {
  stage: number; // 0 to 3
}

const STAGES = [
  {
    icon: Layers,
    title: "Hybrid Retrieval Execution",
    description: "Executing concurrent Qdrant vector search & Elasticsearch BM25 query with Reciprocal Rank Fusion (k=60)...",
    color: "text-blue-500",
    bg: "bg-blue-500/10",
    border: "border-blue-500/25",
  },
  {
    icon: Cpu,
    title: "Cross-Encoder Reranking",
    description: "Computing Cohere/ms-marco rerank scores on top candidate quotes & checking initial confidence...",
    color: "text-indigo-500",
    bg: "bg-indigo-500/10",
    border: "border-indigo-500/25",
  },
  {
    icon: ShieldCheck,
    title: "Contradiction & Groundedness Check",
    description: "Decomposing answer into atomic claims & running NLI pair check (ENTAILMENT vs CONTRADICTION vs NEUTRAL)...",
    color: "text-amber-500",
    bg: "bg-amber-500/10",
    border: "border-amber-500/25",
  },
  {
    icon: Sparkles,
    title: "Synthesizing Verified Answer",
    description: "Applying temporal policy reconciliation & mapping exact chunk IDs to inline citations...",
    color: "text-emerald-500",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/25",
  },
];

export function ThinkingStatus({ stage }: ThinkingStatusProps) {
  const currentStage = STAGES[Math.min(stage, STAGES.length - 1)];
  const Icon = currentStage.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -12, scale: 0.98 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`rounded-2xl border ${currentStage.border} ${currentStage.bg} p-4 shadow-md backdrop-blur-xl max-w-2xl mx-auto`}
    >
      <div className="flex items-center gap-3">
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border ${currentStage.border} bg-background/80 shadow-sm`}>
          <Icon className={`h-5 w-5 ${currentStage.color} animate-spin-slow`} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <span className="font-display text-xs font-bold uppercase tracking-wider text-foreground flex items-center gap-1.5">
              <span>Correction Engine Stage {stage + 1}/4</span>
              <span className="flex h-1.5 w-1.5 rounded-full bg-primary animate-ping" />
            </span>
            <span className="font-mono text-[10px] text-muted-foreground/80">LangGraph StateGraph</span>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={stage}
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.2 }}
              className="mt-0.5"
            >
              <h4 className={`text-sm font-semibold ${currentStage.color}`}>
                {currentStage.title}
              </h4>
              <p className="text-xs text-muted-foreground truncate font-normal">
                {currentStage.description}
              </p>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Progress Dots */}
      <div className="mt-3.5 flex items-center justify-between border-t border-border/30 pt-2.5">
        <div className="flex items-center gap-1.5">
          {STAGES.map((s, idx) => (
            <div
              key={idx}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                idx === stage
                  ? "w-8 bg-primary shadow-sm shadow-primary/50"
                  : idx < stage
                  ? "w-2 bg-emerald-500"
                  : "w-2 bg-border/60"
              }`}
            />
          ))}
        </div>
        <span className="text-[10px] font-semibold text-muted-foreground flex items-center gap-1">
          <RefreshCw className="h-3 w-3 animate-spin text-primary" /> Self-Correcting Loop Active
        </span>
      </div>
    </motion.div>
  );
}
