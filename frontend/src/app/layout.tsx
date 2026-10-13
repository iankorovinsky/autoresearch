import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "../styles.css";

export const metadata: Metadata = {
  title: "Auto Research · Research workspace",
  description:
    "Auto Research — a workspace for autonomous experiments, from hypothesis to faster kernels.",
};

export const viewport: Viewport = { themeColor: "#f8f9f6" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
