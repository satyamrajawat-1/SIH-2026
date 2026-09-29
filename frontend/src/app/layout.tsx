import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SIH v0.1 — AI Human Activity Recognition",
  description:
    "Onboard AI HAR system for tracking astronaut experiment procedures. SIH Problem Statement 26174.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
