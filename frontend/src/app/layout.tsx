import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AtmaNirbhar AI | India-first autonomous driving Multi-Agents AI Intelligence for unstructured roads",
  description:
    "India-first autonomous driving Multi-Agents AI Intelligence for unstructured roads. Real-time perception, acoustic emergency vehicle detection, dynamic replanning, and road hazard intelligence.",
  keywords: [
    "AtmaNirbhar AI",
    "autonomous driving",
    "unstructured roads",
    "Indian roads",
    "multi-agent AI",
    "computer vision",
    "YOLO11",
    "spatial intelligence",
    "time to collision",
    "emergency detection",
    "voice alerts",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
