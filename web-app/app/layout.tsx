import type { Metadata } from "next";
import { Vazirmatn } from "next/font/google";
import "./globals.css";

const vazirmatn = Vazirmatn({
  variable: "--font-vazirmatn",
  subsets: ["arabic", "latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "تشخیص متن هوش مصنوعی | AI Text Detection",
  description:
    "ابزار تشخیص متن تولیدشده توسط هوش مصنوعی برای فارسی. متن خود را وارد کنید و مدل‌های آموزش‌دیده (پیشرفته و کلاسیک) بررسی می‌کنند که متن توسط انسان نوشته شده یا توسط ماشین. شامل مدل ترکیبی، System One برای پاسخ‌های نوع‌بندی‌شده، و گزارش فنی کامل.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fa" dir="rtl" className={`${vazirmatn.variable} dark h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
