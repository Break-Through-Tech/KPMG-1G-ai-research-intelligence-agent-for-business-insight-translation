import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "KPMG AI Assistant" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
