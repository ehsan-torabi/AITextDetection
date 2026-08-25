// app/page.tsx
import { DetectionForm } from "@/components/detection-form";

export default function HomePage() {
  return (
    <main className="min-h-[100dvh] flex flex-col">
      <div className="flex-1 w-full max-w-2xl mx-auto px-4 py-12 sm:py-16 space-y-10">
        {/* Header */}
        <header className="space-y-3">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            تشخیص متن هوش مصنوعی
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-lg">
            متن خود را وارد کنید تا با استفاده از مدل‌های یادگیری ماشین بررسی
            کنیم آیا توسط هوش مصنوعی نوشته شده یا انسان.
          </p>
        </header>

        {/* Form */}
        <DetectionForm />

        {/* Footer */}
        <footer className="text-xs text-muted-foreground pt-4 border-t border-border">
          <p>نتایج این ابزار تخمینی است و دقت قطعی ندارد.</p>
        </footer>
      </div>
    </main>
  );
}