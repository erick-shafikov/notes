## draftMode

включает draft mode, позволяет включить предпросмотр headless CMS

Создаем роут для просмотра

```tsx
// https://<your-site>/api/draft?secret=<token>&slug=<path>
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const secret = searchParams.get("secret");
  const slug = searchParams.get("slug");

  // верификация данных для CMS
  if (secret !== "MY_SECRET_TOKEN" || !slug) {
    return new Response("Invalid token", { status: 401 });
  }

  //получаем данные getPostBySlug - ф-ция для получения данных с CMS
  const post = await getPostBySlug(slug);

  if (!post) {
    return new Response("Invalid slug", { status: 401 });
  }

  // активация
  const draft = await draftMode();
  draft.enable();

  redirect(post.slug);
}
```

используем в компоненте

```tsx
// page that fetches data
import { draftMode } from "next/headers";

async function getData() {
  const { isEnabled } = await draftMode();

  const url = isEnabled
    ? "https://draft.example.com"
    : "https://production.example.com";

  const res = await fetch(url);

  return res.json();
}

export default async function Page() {
  const { title, desc } = await getData();

  return (
    <main>
      <h1>{title}</h1>
      <p>{desc}</p>
    </main>
  );
}
```
