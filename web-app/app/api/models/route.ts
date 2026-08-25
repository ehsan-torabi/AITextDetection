export async function GET() {
  const res = await fetch(`http://0.0.0.0:8000/models`);
  const data = await res.json();
  return Response.json(data);

}
