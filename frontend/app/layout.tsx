import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VOID Operator",
  description: "Frontend control panel for VOID mask generation and inference workflows",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
