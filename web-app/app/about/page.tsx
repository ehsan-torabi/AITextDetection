import Link from "next/link";
import type { Metadata } from "next";
import {
  ArrowRight,
  Cloud,
  Database,
  FileText,
  FlaskConical,
  Layers,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { AboutModels } from "@/components/about-models";

export const metadata: Metadata = {
  title: "روش‌شناسی و داده‌ها | AI Text Detection",
  description:
    "مدل‌ها چگونه آموزش دیده‌اند، داده‌های آموزشی از کجا آمده‌اند، و چرا برخی مدل‌ها هنوز روی متن واقعی اشتباه می‌کنند.",
};

export default function AboutPage() {
  return (
    <main className="min-h-[100dvh] flex flex-col">
      <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-12 sm:py-16 space-y-12">
        {/* Header */}
        <header className="space-y-3">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowRight className="size-3.5" />
            بازگشت به ابزار تشخیص
          </Link>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
            روش‌شناسی: مدل‌ها چگونه آموزش دیده‌اند؟
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-xl">
            این صفحه توضیح می‌دهد داده‌های آموزشی از کجا آمده‌اند، متن فارسی چگونه پیش‌پردازش
            می‌شود، هر مدل با چه روشی آموزش دیده، و — به‌طور صریح — مدل‌ها هنوز کجا اشتباه می‌کنند.
            جزئیات ریاضی و کامل‌تر در{" "}
            <a
              href="/report.pdf"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              گزارش فنی–علمی کامل (PDF)
            </a>{" "}
            موجود است.
          </p>
        </header>

        {/* Data collection */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <Database className="size-4.5 text-primary" />
            داده‌های آموزشی از کجا آمده‌اند؟
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            مجموعهٔ داده از ترکیب متن‌های وبلاگی انسانی و متن‌های تولیدشده توسط هوش مصنوعی ساخته شده
            است. بخش هوش‌مصنوعی داده با یک خط‌لولهٔ گردآوری اختصاصی تولید شده که عمداً بر{" "}
            <strong className="text-foreground">تنوع</strong> تأکید دارد، نه فقط حجم:
          </p>
          <ul className="grid gap-2 sm:grid-cols-2 text-xs">
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">۷۰ مدل زبانی</span>
              <span className="block text-muted-foreground mt-0.5">
                از ۹ ارائه‌دهندهٔ API مختلف (OpenRouter، NVIDIA و دیگران)، خانواده‌های GPT، Claude،
                Gemini/Gemma، DeepSeek، Llama، Qwen و Nemotron
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">۵ سبک نوشتاری</span>
              <span className="block text-muted-foreground mt-0.5">
                مقالهٔ رسمی، پست وبلاگی محاوره‌ای، پرسش‌وپاسخ گام‌به‌گام، متن انتقادی، خلاصهٔ
                ساختاریافته
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">۱٬۴۲۸ موضوع یکتا</span>
              <span className="block text-muted-foreground mt-0.5">
                از علم و فناوری تا اقتصاد، سلامت، فلسفه و هنر
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">دما و top-p تصادفی</span>
              <span className="block text-muted-foreground mt-0.5">
                برای هر نمونه، دما از [۰.۳۰, ۰.۹۵] و top-p از [۰.۸۰, ۱.۰۰] به‌طور تصادفی انتخاب
                می‌شود
              </span>
            </li>
          </ul>
          <Card size="sm" className="bg-primary/5 ring-primary/20">
            <CardContent className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs leading-relaxed">
                <p className="font-medium text-foreground">Persian AI-Generated Text Dataset</p>
                <p className="text-muted-foreground">
                  ۱۰٬۵۴۶ نمونه، با فراداده کامل (مدل، ارائه‌دهنده، سبک، دما، top-p) — به‌صورت عمومی
                  روی Hugging Face منتشر شده است.
                </p>
              </div>
              <a
                href="https://huggingface.co/datasets/ehsantorabi/persian-ai-generated-text"
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/80 transition-colors"
              >
                مشاهده در Hugging Face
              </a>
            </CardContent>
          </Card>
          <p className="text-xs text-muted-foreground">
            گردآوری داده و انتشار مجموعه داده: احسان ترابی فرسانی.
          </p>
        </section>

        {/* Preprocessing + features */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <FlaskConical className="size-4.5 text-primary" />
            پیش‌پردازش و استخراج ویژگی
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            دو خط‌لولهٔ پیش‌پردازش <em>متفاوت</em> به‌کار می‌رود: برای مدل‌های کلاسیک و BiLSTM، متن
            نرمال‌سازی، ریشه‌یابی و توکن‌سازی کامل می‌شود (مناسب برای فضای کیسه‌واژه). برای GPT-2 و
            ParsBERT، تنها نرمال‌سازی سطح کاراکتر انجام می‌شود—بدون ریشه‌یابی—چون این مدل‌ها روی متن
            طبیعی فارسی پیش‌آموزش دیده‌اند و ریشه‌یابی، ورودی را از توزیع پیش‌آموزش آن‌ها دور
            می‌کند.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            برای مدل‌های کلاسیک، علاوه بر بردار TF-IDF (یک‌گرم تا سه‌گرم، سقف ۱۵۰٬۰۰۰ ویژگی)، چهار
            ویژگی سبک‌شناسی دست‌ساز نیز محاسبه و به بردار ورودی الحاق می‌شود:
          </p>
          <ul className="grid gap-2 sm:grid-cols-2 text-xs">
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">آنتروپی کاراکتری</span>
              <span className="block text-muted-foreground mt-0.5">
                تنوع کاراکتری متن، نرمال‌شده بین ۰ تا ۱
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">تنوع طول جمله</span>
              <span className="block text-muted-foreground mt-0.5">
                ضریب تغییرات طول جمله‌های متوالی
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">تکرار سه‌گرم</span>
              <span className="block text-muted-foreground mt-0.5">
                نسبت سه‌گرم‌های واژگانی تکرارشونده
              </span>
            </li>
            <li className="rounded-lg bg-muted/50 px-3 py-2">
              <span className="font-semibold text-foreground">انسجام میان‌جمله‌ای</span>
              <span className="block text-muted-foreground mt-0.5">
                شباهت کسینوسی TF-IDF میان جمله‌های متوالی
              </span>
            </li>
          </ul>
        </section>

        {/* Training methodology */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <Layers className="size-4.5 text-primary" />
            روش‌شناسی آموزش
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            هر مدل کلاسیک (SVM، رگرسیون لجستیک، بیز ساده، جنگل تصادفی، گرادیان بوستینگ) با جست‌وجوی
            هایپرپارامتر واقعی (اعتبارسنجی متقاطع k-تاخورده) تنظیم می‌شود. داده به سه بخش
            آموزش/اعتبارسنجی/آزمون تفکیک می‌شود—با نسبت طبقه‌بندی‌شدهٔ ثابت—و مجموعهٔ آزمون تنها
            یک‌بار، در پایان، برای هر مدل لمس می‌شود. مدل‌های یادگیری عمیق (BiLSTM، GPT-2، و
            ParsBERT) بهترین نقطهٔ توقف خود را بر پایهٔ امتیاز F1 روی مجموعهٔ اعتبارسنجی—نه آزمون—با
            توقف زودهنگام تعیین می‌کنند.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            در کنار این مدل‌ها، یک لایهٔ تصمیم «سیستم-یک» نیز سرو می‌شود: به‌جای یک عدد، سه پاسخ
            نوع‌بندی‌شده—انتخاب منبع، امتیاز ماشین‌مانندبودن روی طیف پنج‌سطحی و احتمال نیاز به
            بازبینی انسانی—در یک گذر رو به جلو برمی‌گرداند. بدنهٔ آن همان رمزگذار ParsBERT
            فاین‌تیون‌شده است و چون مجموعه داده فقط برچسب دودویی دارد، هدف‌های نرم آموزشش از
            دسته‌بند GPT-2 فاین‌تیون‌شده—مدلی با بیشترین وزن در رأی‌گیری ترکیبی—تقطیر شده است.
          </p>
        </section>

        {/* Online System One mode */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <Cloud className="size-4.5 text-primary" />
            حالت آنلاین System One
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            علاوه بر مدل‌های محلی، می‌توانید همان متن را با مدل Jev سرویس{" "}
            <a
              href="https://docs.typesafe.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-4"
            >
              TypeSafe
            </a>{" "}
            — نخستین مدل «سیستم-یک» تجاری — تحلیل کنید. این مدل به‌جای تولید متن، همان سه پاسخ
            نوع‌بندی‌شدهٔ محلی (انتخاب منبع، امتیاز ماشین‌مانندبودن و احتمال نیاز به بازبینی) را
            برمی‌گرداند؛ به همین دلیل خروجی هر دو در حالت «مقایسه هر دو» کنار هم و سؤال‌به‌سؤال قابل
            سنجش است.
          </p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            تنظیمات اتصال (کلید API، آدرس سرویس و نام مدل) در مرورگر خودتان — در{" "}
            <code className="mx-1 text-xs">localStorage</code> — ذخیره می‌شود و تنها هنگام تحلیل به
            سرویس منتقل می‌گردد. یک نکتهٔ صادقانه: زبان اصلی آموزش Jev انگلیسی است و دقت آن روی متن
            فارسی — مانند مدل‌های محلی — محدودیت دارد؛ مقایسهٔ دو رویکرد را به‌عنوان سیگنال ببینید،
            نه حکم نهایی.
          </p>
        </section>

        {/* Honest error analysis */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <TriangleAlert className="size-4.5 text-amber-500" />
            کجا مدل‌ها هنوز اشتباه می‌کنند؟
          </h2>
          <Card size="sm" className="bg-amber-500/5 ring-amber-500/20">
            <CardContent className="space-y-2 text-xs leading-relaxed text-muted-foreground">
              <p>
                بررسی هدفمند نشان داد <strong className="text-foreground">BiLSTM</strong> — که
                embedding خود را از ابتدا روی تنها ~۱۷٬۰۰۰ سند می‌آموزد، بدون هیچ دانش زبانی پیشین —
                در برابر سبک‌های نوشتاری خارج از توزیع آموزشی (مثلاً متن هوش‌مصنوعی با لحن
                محاوره‌ای) آسیب‌پذیر است. در یک آزمون، یک متن محاوره‌ای که واقعاً توسط هوش مصنوعی
                نوشته شده بود، با اطمینان ۹۹.۹٪ «انسانی» طبقه‌بندی شد.
              </p>
              <p>
                نکتهٔ مهم: حتی <strong className="text-foreground">GPT-2</strong> (با وجود دانش
                زبانی پیش‌آموزش‌دیده) روی همان نمونه اشتباه کرد. این نشان می‌دهد مشکل صرفاً یک اشکال
                در معماری BiLSTM نیست، بلکه محدودیت تنوع سبکی دادهٔ آموزشی است.
              </p>
              <p>
                به همین دلیل، <strong className="text-foreground">ParsBERT</strong> — یک رمزگذار
                دوسویهٔ فارسی از پیش‌آموزش‌دیده، مشابه در روح به GPT-2 اما با معماری مناسب‌تر برای
                طبقه‌بندی — به‌عنوان مدل یادگیری عمیق چهارم آموزش دیده و به سرویس افزوده شده است.
                همین رمزگذار ستون فقرات لایهٔ تصمیم «سیستم-یک» نیز هست. جزئیات کامل این تحلیل در
                گزارش PDF آمده است.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Live models */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-lg font-bold">
            <Sparkles className="size-4.5 text-primary" />
            مدل‌ها و معیارهای زنده
          </h2>
          <p className="text-sm text-muted-foreground leading-relaxed">
            فهرست زیر مستقیماً از سرور بک‌اند خوانده می‌شود، پس همیشه با
            <code className="mx-1 text-xs">models.json</code>
            به‌روز است.
          </p>
          <AboutModels />
        </section>

        {/* Footer */}
        <footer className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground pt-4 border-t border-border">
          <p>گزارش فنی–علمی پروژه — احسان ترابی فرسانی</p>
          <a
            href="/report.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-primary hover:underline underline-offset-4"
          >
            <FileText className="size-3.5" />
            دانلود گزارش کامل (PDF)
          </a>
        </footer>
      </div>
    </main>
  );
}
