// app/page.tsx
import Link from "next/link";
import { FileText } from "lucide-react";
import { DetectionForm } from "@/components/detection-form";

export const metadata = {
  title: "تشخیص متن هوش مصنوعی",
  description:
    "متن خود را وارد کنید تا مدل‌های آموزش‌دیده بررسی کنند آیا متن توسط انسان نوشته شده یا توسط هوش مصنوعی تولید شده است.",
};

export default function HomePage() {
  return (
    <main className="min-h-[100dvh] flex flex-col">
      <div className="flex-1 w-full max-w-2xl mx-auto px-4 py-12 sm:py-16 space-y-10">
        {/* Student-project warning */}
        <section
          role="note"
          aria-label="هشدار پروژه دانشجویی"
          className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-4 text-xs text-amber-700 leading-relaxed dark:border-amber-400/30 dark:text-amber-300"
        >
          <strong className="font-bold block mb-1">
            هشدار — پروژه پایان‌ترم دانشجویی
          </strong>
          این ابزار یک پروژه پایان‌ترم دانشجویی است که توسط{" "}
          <strong className="font-bold">
            احسان ترابی فارسانی (Ehsan Torabi Farsani)
          </strong>{" "}
          توسعه یافته است. خروجی‌های آن تخمینی هستند، دقت قطعی ندارند و برای تصمیمات
          مهم یا استفادهٔ تجاری مناسب نیستند.
        </section>

        {/* Header */}
        <header className="space-y-3">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            تشخیص متن هوش مصنوعی
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-lg">
            متن خود را وارد کنید تا مدل‌های آموزش‌دیده بررسی کنند آیا متن توسط انسان
            نوشته شده یا توسط هوش مصنوعی تولید شده است. نتیجه تخمینی است و دقت قطعی
            ندارد.
          </p>
        </header>

        {/* Form */}
        <DetectionForm />

        {/* Footer */}
        <footer className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground pt-4 border-t border-border">
          <p>نتیجه تخمینی است — دقت قطعی ندارد.</p>
          <Link
            href="/about"
            className="text-primary hover:underline underline-offset-4"
          >
            مدل‌ها چگونه آموزش دیده‌اند؟
          </Link>
          <a
            href="https://ehsantorabi.ir"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-primary hover:underline underline-offset-4"
          >
            <FileText className="size-3.5" aria-hidden="true" />
            وب‌سایت
          </a>
        </footer>
      </div>
    </main>
  );
}
