import { backendUrl } from "@/lib/backend-url";

export async function GET() {
  const res = await fetch(backendUrl("/models"));
  const data = await res.json();
  return Response.json(data);
}
