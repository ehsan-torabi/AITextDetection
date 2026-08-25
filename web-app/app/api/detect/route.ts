import type { DetectionRequest } from "@/lib/types";

export async function POST(request: Request) {
  const body: DetectionRequest = await request.json();

  if (!body.text || body.text.trim().length < 20) {
    return Response.json(
      { error: "متن ارسال شده باید حداقل ۲۰ کاراکتر باشد." },
      { status: 400 }
    );
  }

  if (!body.modelId) {
    return Response.json(
      { error: "لطفاً یک مدل انتخاب کنید." },
      { status: 400 }
    );
  }


  const res = await fetch(`http://0.0.0.0:8000/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return Response.json(data);

}
