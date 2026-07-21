"use client";

import React, { useState, useEffect } from "react";
import { StatCards } from "@/components/evaluation/stat-cards";
import { BenchmarkCharts } from "@/components/evaluation/benchmark-charts";
import { type EvaluationMetric } from "@/lib/types";
import { apiClient } from "@/lib/api";
import { 
  ShieldCheck, 
  CheckCircle2, 
  XCircle, 
  Download, 
  Search, 
  Sparkles,
  Layers
} from "lucide-react";
import { toast } from "sonner";

export default function EvaluationPage() {
  const [metrics, setMetrics] = useState<EvaluationMetric[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>("ALL");
  const [search, setSearch] = useState("");

  useEffect(() => {
    const data = apiClient.getEvaluationMetrics();
    setMetrics(data);
    setIsLoading(false);
  }, []);

  const filteredMetrics = metrics.filter((m) => {
    const matchesCat = selectedCategory === "ALL" || m.category === selectedCategory;
    const matchesSearch = m.question.toLowerCase().includes(search.toLowerCase()) || m.question_id.toLowerCase().includes(search.toLowerCase());
    return matchesCat && matchesSearch;
  });

  const handleExport = () => {
    const jsonStr = JSON.stringify(metrics, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "compass_rag_evaluation_benchmark_results.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success("Benchmark report exported as JSON!");
  };

  if (isLoading) {
    return <div className="p-8 text-center text-muted-foreground animate-pulse">Loading evaluation benchmarks...</div>;
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Executive Overview Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-2xl border border-border/60 bg-gradient-to-r from-card via-card/90 to-emerald-500/5 p-6 shadow-sm backdrop-blur-xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold text-emerald-500 border border-emerald-500/30 uppercase tracking-wider">
              Benchmark Studio
            </span>
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Sparkles className="h-3.5 w-3.5 text-primary" /> 15 Golden Dataset Questions
            </span>
          </div>
          <h2 className="font-display text-2xl font-bold tracking-tight text-foreground">
            Evaluation & Hallucination Benchmarking
          </h2>
          <p className="text-xs text-muted-foreground max-w-2xl">
            Comparing our Baseline RAG pipeline against the 9-node Self-Correcting RAG engine. Real-time metrics compute factual groundedness, NLI contradiction resolution, and programmatic citation correctness.
          </p>
        </div>

        <button
          onClick={handleExport}
          className="flex items-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-xs font-bold text-primary-foreground hover:bg-primary/90 shadow-md shadow-primary/25 transition-all shrink-0"
        >
          <Download className="h-4 w-4" />
          <span>Export Benchmark Report (JSON)</span>
        </button>
      </div>

      {/* Executive Reduction Stat Cards */}
      <StatCards metrics={metrics} />

      {/* Recharts Bar & Radar Comparison Studio */}
      <BenchmarkCharts metrics={metrics} />

      {/* Detailed Breakdown Table */}
      <div className="rounded-2xl border border-border/80 bg-card/90 shadow-lg backdrop-blur-xl overflow-hidden">
        {/* Table Toolbar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 border-b border-border/60 px-6 py-4">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-primary" />
            <h3 className="font-display text-sm font-bold text-foreground">
              Golden Dataset Question Audit Log
            </h3>
            <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-mono text-muted-foreground">
              {filteredMetrics.length} / {metrics.length}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Category select */}
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="rounded-xl border border-border/60 bg-background px-3 py-1.5 text-xs text-foreground focus:border-primary focus:outline-none"
            >
              <option value="ALL">All Categories</option>
              <option value="Directly Answerable">Directly Answerable</option>
              <option value="OCR Dependent">OCR Dependent</option>
              <option value="Contradictory Policy">Contradictory Policy</option>
              <option value="Ambiguous">Ambiguous</option>
              <option value="Unanswerable">Unanswerable</option>
            </select>

            {/* Search */}
            <div className="relative w-48">
              <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search question..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full rounded-xl border border-border/60 bg-background py-1.5 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Table View */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border/60 bg-background/50 font-semibold uppercase tracking-wider text-muted-foreground">
              <tr>
                <th className="py-3.5 pl-6 pr-3">ID</th>
                <th className="py-3.5 px-3">Category</th>
                <th className="py-3.5 px-3 min-w-[240px]">Golden Question</th>
                <th className="py-3.5 px-3 text-center">Baseline RAG</th>
                <th className="py-3.5 px-3 text-center">Self-Correcting RAG</th>
                <th className="py-3.5 pl-3 pr-6 text-right">Latency Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/40">
              {filteredMetrics.map((m) => (
                <tr key={m.question_id} className="hover:bg-accent/40 transition-colors">
                  <td className="py-3.5 pl-6 pr-3 font-mono font-bold text-primary">
                    {m.question_id}
                  </td>
                  <td className="py-3.5 px-3">
                    <span className="inline-flex rounded-full bg-secondary px-2.5 py-0.5 text-[11px] font-medium text-foreground">
                      {m.category}
                    </span>
                  </td>
                  <td className="py-3.5 px-3 font-medium text-foreground">
                    {m.question}
                  </td>
                  <td className="py-3.5 px-3 text-center">
                    <div className="flex items-center justify-center gap-1.5">
                      {m.baseline_hallucinated ? (
                        <span className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-bold text-rose-500">
                          <XCircle className="h-3 w-3" /> Hallucinated
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-bold text-emerald-500">
                          <CheckCircle2 className="h-3 w-3" /> Grounded
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3.5 px-3 text-center">
                    <div className="flex items-center justify-center gap-1.5">
                      {m.corrected_hallucinated ? (
                        <span className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-bold text-rose-500">
                          <XCircle className="h-3 w-3" /> Hallucinated
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-bold text-emerald-500 shadow-sm">
                          <ShieldCheck className="h-3 w-3" /> 100% Verified
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3.5 pl-3 pr-6 text-right font-mono text-muted-foreground">
                    <span className="text-foreground font-semibold">{m.corrected_latency_ms}ms</span>
                    <span className="text-[10px] text-muted-foreground/80 ml-1">
                      (+{m.corrected_latency_ms - m.baseline_latency_ms}ms)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
