"use client";

import React, { useState } from "react";
import { QueryInput } from "@/components/chat/query-input";
import { ThinkingStatus } from "@/components/chat/thinking-status";
import { ResponseCard } from "@/components/chat/response-card";
import { CitationSheet } from "@/components/chat/citation-sheet";
import { type Citation, type QueryResponse } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { Sparkles, ShieldCheck, RefreshCw, Cpu } from "lucide-react";

interface ChatTurn {
  question: string;
  response: QueryResponse;
}

export default function ChatPage() {
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [thinkingStage, setThinkingStage] = useState(0);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);

  const handleSendQuery = async (question: string) => {
    setIsLoading(true);
    setThinkingStage(0);

    // Simulate animated state transitions through the 4 LangGraph stages
    const stageInterval = setInterval(() => {
      setThinkingStage((prev) => {
        if (prev < 3) return prev + 1;
        return prev;
      });
    }, 450);

    try {
      const response = await apiClient.query(question);
      clearInterval(stageInterval);
      setHistory((prev) => [{ question, response }, ...prev]);
    } catch (err) {
      clearInterval(stageInterval);
      console.error("Query failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFollowupReply = (replyText: string) => {
    handleSendQuery(replyText);
  };

  return (
    <div className="flex flex-col h-full space-y-6 pb-12">
      {/* Executive Centerpiece Banner */}
      {history.length === 0 && !isLoading && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-border/60 bg-gradient-to-b from-card via-card/95 to-primary/5 p-10 text-center shadow-md backdrop-blur-xl">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-indigo-600 text-primary-foreground shadow-lg shadow-primary/30 mb-5">
            <Sparkles className="h-8 w-8 animate-pulse" />
          </div>

          <h2 className="font-display text-2xl md:text-3xl font-bold tracking-tight text-foreground max-w-2xl">
            Self-Correcting RAG with NLI Contradiction & Atomic Groundedness Check
          </h2>
          <p className="mt-2 text-xs md:text-sm text-muted-foreground max-w-xl leading-relaxed">
            Experience our 9-node LangGraph StateGraph engine. Ask complex policy questions across your enterprise knowledgebase with full Reciprocal Rank Fusion, Cohere Reranking, and zero-hallucination verification.
          </p>

          <div className="mt-6 flex flex-wrap items-center justify-center gap-3 text-xs font-semibold">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-emerald-500">
              <ShieldCheck className="h-3.5 w-3.5" /> VERIFIED Groundedness
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-amber-500">
              <RefreshCw className="h-3.5 w-3.5" /> CLARIFICATION Router
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-rose-500/30 bg-rose-500/10 px-3 py-1 text-rose-500">
              <Cpu className="h-3.5 w-3.5" /> LOW CONFIDENCE Abstention
            </span>
          </div>
        </div>
      )}

      {/* Query Input Box Component */}
      <QueryInput onSend={handleSendQuery} isLoading={isLoading} />

      {/* Active Thinking State Indicator */}
      {isLoading && (
        <div className="py-4">
          <ThinkingStatus stage={thinkingStage} />
        </div>
      )}

      {/* Conversational Feed */}
      <div className="space-y-6 flex-1 overflow-y-auto">
        {history.map((turn, idx) => (
          <ResponseCard
            key={idx}
            question={turn.question}
            response={turn.response}
            onCitationClick={(cit) => setSelectedCitation(cit)}
            onFollowupReply={handleFollowupReply}
          />
        ))}
      </div>

      {/* Citation Slide-Out Sheet Inspector */}
      <CitationSheet
        citation={selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
}
