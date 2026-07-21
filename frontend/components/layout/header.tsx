"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { 
  Building2, 
  Sun, 
  Moon, 
  Monitor, 
  Activity, 
  ChevronRight, 
  Lock
} from "lucide-react";
import { cn } from "@/lib/utils";

export function Header() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();

  const getPageTitle = (path: string) => {
    if (path === "/chat" || path === "/") return "Self-Correcting Query Center";
    if (path.startsWith("/library")) return "Enterprise Document Library";
    if (path.startsWith("/review")) return "Manual OCR Correction Queue";
    if (path.startsWith("/evaluation")) return "Evaluation & Benchmarking Studio";
    return "Dashboard";
  };

  return (
    <header className="sticky top-0 z-20 flex h-16 w-full items-center justify-between border-b border-border/60 bg-background/80 px-6 backdrop-blur-xl shadow-sm">
      {/* Left: Breadcrumbs & Page Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <span>Compass Platform</span>
          <ChevronRight className="h-3.5 w-3.5 text-border" />
        </div>
        <h1 className="font-display text-lg font-bold tracking-tight text-foreground flex items-center gap-2">
          {getPageTitle(pathname)}
        </h1>
      </div>

      {/* Right: Tenant Scope Selector, Status Indicator, & Theme Toggle */}
      <div className="flex items-center gap-3">
        {/* Tenant Isolation Selector */}
        <div className="flex items-center gap-2 rounded-xl border border-border/60 bg-card px-3 py-1.5 text-xs shadow-sm">
          <Building2 className="h-4 w-4 text-primary" />
          <span className="text-muted-foreground font-medium">Tenant Scope:</span>
          <span className="font-semibold text-foreground flex items-center gap-1" title="Strict Tenant RBAC Isolation Active">
            Enterprise HQ (Tenant A)
            <Lock className="h-3 w-3 text-emerald-500 ml-0.5" />
          </span>
        </div>

        {/* Live System Status Badge */}
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/25 bg-emerald-500/10 px-3 py-1.5 text-xs font-semibold text-emerald-500 shadow-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <Activity className="h-3.5 w-3.5" />
          <span>API Gateway: Online</span>
        </div>

        {/* Theme Toggle (Light / Dark / System) */}
        <div className="flex items-center rounded-xl border border-border/60 bg-card p-1 shadow-sm">
          <button
            onClick={() => setTheme("light")}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              theme === "light" ? "bg-accent text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
            title="Light theme"
          >
            <Sun className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setTheme("dark")}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              theme === "dark" ? "bg-accent text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
            title="Dark theme"
          >
            <Moon className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setTheme("system")}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              theme === "system" ? "bg-accent text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
            title="System theme"
          >
            <Monitor className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
}
