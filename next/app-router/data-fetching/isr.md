ISR

- кеш дял isr общий для всех
- Middleware не будет выполняться для запросов ISR
- если у fetch revalidate === 0 или no-store то путь будет динамический

```tsx
//app/blog/[id]/page.tsx
interface Post {
  id: string;
  title: string;
  content: string;
}

//данные кешируются каждые 60 секунд, в фоновом режиме будет генерится новая
export const revalidate = 60;

//сгенерирует данные в build time, потом при запросе
export const dynamicParams = true; // если false, то вернет 404 для не сгенерированных

//достанет все id
export async function generateStaticParams() {
  const posts: Post[] = await fetch("https://api.vercel.app/blog").then((res) =>
    res.json()
  );
  return posts.map((post) => ({
    id: String(post.id),
  }));
}

//для каждой отдельной страницы
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post: Post = await fetch(`https://api.vercel.app/blog/${id}`).then(
    (res) => res.json()
  );
  return (
    <main>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </main>
  );
}
```

дял точечной ревалидации данных, unstable_cache

```ts
"use server";

import { revalidatePath } from "next/cache";

export async function createPost() {
  // ре-валидация по тегу
  revalidatePath("/posts");
}

export default async function Page() {
  const data = await fetch("https://api.vercel.app/blog", {
    next: { tags: ["posts"] },
  });
  const posts = await data.json();
  // ...
}
```

# дебаг

собрать проект с настройкой

```js
module.exports = {
  logging: {
    fetches: {
      fullUrl: true,
    },
  },
};
```

```
NEXT_PRIVATE_DEBUG_CACHE=1
```
