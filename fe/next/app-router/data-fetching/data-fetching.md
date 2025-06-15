Варианты загрузки данных:

1. На сервере с помощью fetch (в route-handlers, серверных компонентах, server-action)
2. На сервере с мощью сторонних библиотек
3. На клиенте с помощью Rout-Handler
4. На клиенте с мощью сторонних библиотек

<!-- Кэширование ---------------------------------------------------->

# Кэширование

- не кешируется по умолчанию

```tsx
//что бы не кешировалось, предотвратить предварительную отрисовку
export const dynamic = "force-dynamic";
```

- формируется при build, обновить с помощью ISR
- вызов cookies, headers, searchParams заставит страницу отображаться динамически, тогда ненужно force-dynamic
- unstable_cache для кешированияъ

```tsx
//обеспечит кеширование на час
import { unstable_cache } from "next/cache";
import { db, posts } from "@/lib/db";

const getPosts = unstable_cache(
  async () => {
    return await db.select().from(posts);
  },
  ["posts"],
  { revalidate: 3600, tags: ["posts"] }
);

export default async function Page() {
  const allPosts = await getPosts();

  return (
    <ul>
      {allPosts.map((post) => (
        <li key={post.id}>{post.title}</li>
      ))}
    </ul>
  );
}
```

кеширование

```tsx
import { notFound } from "next/navigation";

interface Post {
  id: string;
  title: string;
  content: string;
}

async function getPost(id: string) {
  const res = await fetch(`https://api.vercel.app/blog/${id}`, {
    // cache: 'force-cache' - позволит закешировать данные
    cache: "force-cache",
  });
  const post: Post = await res.json();
  if (!post) notFound();
  return post;
}

export async function generateStaticParams() {
  const posts = await fetch("https://api.vercel.app/blog", {
    // cache: 'force-cache' - позволит закешировать данные
    cache: "force-cache",
  }).then((res) => res.json());

  return posts.map((post: Post) => ({
    id: String(post.id),
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post = await getPost(id);

  return {
    title: post.title,
  };
}

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const post = await getPost(id);

  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </article>
  );
}
```

не для fetch запросов - cache из react

- Мемоизация применяется только к GETметоду в fetch запросах.
- Он применяется к fetch запросам в generateMetadata, generateStaticParams
- Это не относится к fetch запросам в обработчиках маршрутов

```tsx
import { cache } from "react";
import { db, posts, eq } from "@/lib/db"; // Example with Drizzle ORM
import { notFound } from "next/navigation";

export const getPost = cache(async (id) => {
  const post = await db.query.posts.findFirst({
    where: eq(posts.id, parseInt(id)),
  });

  if (!post) notFound();
  return post;
});
```

запросы могут зависеть друг от друга, тогда можно использовать Promise.all()

```tsx
//@/components/Item
//пример предварительной загрузки с помощью функции preload
import { getItem } from "@/utils/get-item";

export const preload = (id: string) => {
  void getItem(id);
};
export default async function Item({ id }: { id: string }) {
  const result = await getItem(id);
  // ...
}
```

```tsx
import Item, { preload, checkIsAvailable } from "@/components/Item";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // starting loading item data
  preload(id);
  // perform another asynchronous task
  const isAvailable = await checkIsAvailable();

  return isAvailable ? <Item id={id} /> : null;
}
```

## use + suspense

```tsx
import Posts from '@/app/ui/posts
import { Suspense } from 'react'

export default function Page() {
  // Don't await the data fetching function
  const posts = getPosts()

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Posts posts={posts} />
    </Suspense>
  )
}
```

в клиентском компоненте

```tsx
"use client";
import { use } from "react";

export default function Posts({
  posts,
}: {
  posts: Promise<{ id: string; title: string }[]>;
}) {
  const allPosts = use(posts);

  return (
    <ul>
      {allPosts.map((post) => (
        <li key={post.id}>{post.title}</li>
      ))}
    </ul>
  );
}
```
