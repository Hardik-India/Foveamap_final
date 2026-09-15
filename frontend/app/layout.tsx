import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FoveaMap | LiDAR Perception Workspace",
  description: "Adaptive variable-resolution 2.5D LiDAR mapping, simulation and evaluation. SIH 2026.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
