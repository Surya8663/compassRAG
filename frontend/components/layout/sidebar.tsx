"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  Compass, 
  MessageSquare, 
  FolderOpen, 
  CheckSquare, 
  BarChart3, 
  ChevronLeft, 
  ChevronRight, 
  ShieldCheck,
  Cpu,
  Sparkles
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  {
    title: "Self-Correcting Chat",
    href: "/chat",
    icon: MessageSquare,
    description: "Hybrid RAG with live verification & RRF",
    badge: "AI Core",
  },
  {
    title: "Document Library",
    href: "/library",
    icon: FolderOpen,
    description: "Ingestion queue & OCR pipelines",
  },
  {
    title: "Manual OCR Review",
    href: "/review",
    icon: CheckSquare,
    description: "Resolve flagged low-confidence scans",
    badgeCount: 2,
  },
  {
    title: "Evaluation Benchmarks",
    href: "/evaluation",
    icon: BarChart3,
    description: "Baseline vs Corrected metrics comparison",
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "relative flex flex-col border-r border-border/60 bg-card/80 backdrop-blur-xl transition-all duration-300 z-30 select-none",
        collapsed ? "w-20" : "w-72"
      )}
    >
      {/* Brand Header */}
      <div className="flex h-16 items-center justify-between px-4 border-b border-border/40">
        <Link href="/" className="flex items-center gap-3 overflow-hidden">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-indigo-600 text-primary-foreground shadow-lg shadow-primary/25">
            <Compass className="h-5 w-5 animate-spin-slow" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="font-display text-base font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                Compass RAG
              </span>
              <span className="text-[10px] uppercase font-semibold tracking-wider text-primary flex items-center gap-1">
                <Sparkles className="h-2.5 w-2.5" /> Enterprise v2.0
              </span>
            </div>
          )}
        </Link>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex h-7 w-7 items-center justify-center rounded-lg border border-border/60 bg-background/50 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {/* Navigation Feed */}
      <div className="flex-1 overflow-y-auto py-6 px-3 space-y-1.5">
        {!collapsed && (
          <div className="px-3 pb-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
            Platform Workflows
          </div>
        )}

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "group relative flex items-center gap-3.5 rounded-xl px-3.5 py-3 transition-all duration-200 font-medium text-sm",
                isActive
                  ? "bg-primary/10 text-primary shadow-sm shadow-primary/5 border border-primary/20"
                  : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
              )}
            >
              {isActive && (
                <div className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r-full bg-primary shadow-[0_0_8px_rgba(99,102,241,0.6)]" />
              )}

              <Icon
                className={cn(
                  "h-5 w-5 shrink-0 transition-transform group-hover:scale-110 duration-200",
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                )}
              />

              {!collapsed && (
                <div className="flex flex-1 items-center justify-between min-w-0">
                  <div className="flex flex-col min-w-0">
                    <span className="truncate">{item.title}</span>
                    <span className="text-[11px] text-muted-foreground/80 truncate font-normal">
                      {item.description}
                    </span>
                  </div>

                  {item.badge && (
                    <span className="ml-2 rounded-md bg-primary/20 px-2 py-0.5 text-[10px] font-semibold text-primary border border-primary/30">
                      {item.badge}
                    </span>
                  )}
                  {item.badgeCount && (
                    <span className="ml-2 flex h-5 w-5 items-center justify-center rounded-full bg-amber-500/20 text-[11px] font-bold text-amber-500 border border-amber-500/30 animate-pulse">
                      {item.badgeCount}
                    </span>
                  )}
                </div>
              )}
            </Link>
          );
        })}
      </div>

      {/* Footer / System Status Card */}
      {!collapsed && (
        <div className="p-4 border-t border-border/40">
          <div className="rounded-xl border border-border/60 bg-background/50 p-3.5 space-y-2.5 shadow-sm">
            <div className="flex items-center justify-between text-xs font-semibold text-foreground">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="h-4 w-4 text-emerald-500" />
                RBAC Enforced
              </span>
              <span className="inline-flex items-center rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-500 border border-emerald-500/20">
                Active
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <Cpu className="h-3.5 w-3.5 text-primary" /> Qdrant + ES
              </span>
              <span className="font-mono text-[10px]">Hybrid RRF k=60</span>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
