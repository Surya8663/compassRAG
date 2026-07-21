import type { Metadata } from "next";
import { Inter, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import React from "react";
import { ThemeProvider } from "@/components/theme-provider";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { Toaster } from "sonner";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Compass RAG | Enterprise Document Intelligence & Hybrid RAG",
  description:
    "Self-correcting RAG platform with Qdrant vector retrieval, Elasticsearch BM25, Reciprocal Rank Fusion, Cross-Encoder reranking, and NLI contradiction detection.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${plusJakartaSans.variable}`}>
      <body className="font-sans min-h-screen bg-background antialiased flex flex-col">
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex h-screen overflow-hidden">
            {/* Sidebar Navigation */}
            <Sidebar />

            {/* Main Content Area */}
            <div className="flex flex-1 flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-y-auto bg-background/50 p-6 md:p-8">
                <div className="mx-auto max-w-7xl h-full">{children}</div>
              </main>
            </div>
          </div>

          {/* Toast Notification System */}
          <Toaster
            position="top-right"
            theme="system"
            richColors
            closeButton
            toastOptions={{
              className: "rounded-xl border border-border/80 backdrop-blur-xl shadow-lg",
            }}
          />
        </ThemeProvider>
      </body>
    </html>
  );
}
