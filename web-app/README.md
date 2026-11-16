# Web app — AI Text Detection (Persian)

> **هشدار — پروژه پایان‌ترم دانشجویی:** این ابزار در cadrul پروژه پایان‌ترم دانشجویی توسط احسان ترابی فرسانی (Ehsan Torabi Farsani) توسعه یافته است. خروجی‌های آن تخمینی هستند، دقت قطعی ندارند و برای تصمیمات مهم یا استفادهٔ تجاری مناسب نیستند.

Next.js front end for the Persian AI-generated text detection tool. It talks to the FastAPI serving stack (`aitextd.api`, port 8000) for local model detections and proxies the online System One (TypeSafe) requests.

## Layout

```
web-app/
app/                  Next.js App Router pages and API routes
  page.tsx            home: detection form + result
  about/              methodology, data, and model docs
  api/detect/         proxy → backend POST /detect
  api/models/         proxy → backend GET /models
  api/typesafe/detect/ server-side proxy for online System One detection
components/           detection UI (form, result, model stats, comparison, provider settings)
components/ui/        shadcn-style primitives (button, input, textarea, card, ...)
lib/                  types, models hook, provider settings (localStorage), utils
public/               static assets (report.pdf, icons)
```

## Routes

| Route | Purpose |
|---|---|
| `/` | Detection form: choose mode (local / online / compare), pick a model, submit text, view results |
| `/about` | Methodology: training data, preprocessing, model training, System One typed answers, error analysis, live model table |
| `/api/detect` | Server proxy to the local backend `POST /detect` |
| `/api/models` | Server proxy to the backend `GET /models` |
| `/api/typesafe/detect` | Server-side proxy to the online System One provider |

## Getting started

```bash
cd web-app
pnpm install
pnpm dev        # http://localhost:3000
```

The front end expects the Python API on `http://0.0.0.0:8000` (see the repo root `server.py`). Start it first:

```bash
# repo root
.venv/bin/python server.py --download-online
```

Then `pnpm dev` in `web-app/`.

## Online System One mode

The online mode calls TypeSafe's Jev model through the `/api/typesafe/detect` server proxy. Connection settings (base URL, model, API key) live in the browser's `localStorage` only — the key is forwarded with each request and never stored on the server. Fill them in via the "سرویس آنلاین System One" panel before running an online or compare detection.

## Build & lint

```bash
pnpm build      # production build
pnpm start      # run production server
pnpm lint       # ESLint
```

## Notes

- Persian UI (RTL, Vazirmatn font). Copy is in Persian; English terms kept where they match the backend (model ids, metric names).
- Model metrics come from `models.json` served by the backend; without a running backend the model list is empty.
- This is a detection tool, not a marketing landing page — keep copy functional, concise, and in Persian. Avoid generic AI-marketing phrasing.

## License

MIT — see the repo root `LICENSE`.
