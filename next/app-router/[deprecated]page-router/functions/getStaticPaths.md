# getStaticPaths (на сервере, для динамических)

если экспортировать функцию getStaticPaths из компонента, то она загрузит все динамические роуты

```tsx
import type {
  InferGetStaticPropsType,
  GetStaticProps,
  GetStaticPaths,
} from "next";

type Repo = {
  name: string;
  stargazers_count: number;
};

export const getStaticPaths = (async () => {
  return {
    paths: [
      {
        params: {
          name: "next.js",
        },
      }, // See the "paths" section below
    ],
    fallback: true, // false or "blocking"
  };
}) satisfies GetStaticPaths;

export const getStaticProps = (async (context) => {
  const res = await fetch("https://api.github.com/repos/vercel/next.js");
  const repo = await res.json();
  return { props: { repo } };
}) satisfies GetStaticProps<{
  repo: Repo;
}>;

export default function Page({
  repo,
}: InferGetStaticPropsType<typeof getStaticProps>) {
  return repo.stargazers_count;
}
```

должен возвращать объект вида

```ts
// pages/posts/[nameOfId].js
type ReturnTypeOfGetStaticPaths = {
  path: [
    { params: { ["nameOfId"]: "id1" } },
    { params: { ["nameOfId"]: "id2" } }
    //для строк вида [some]/[nested]/[params].js
    { params: { ["some", "nested", "params"]: "id2" } }
  ];
  fallback: false; //все совпадения, которые не нашлись в path отправят на 404, на сервере сгенерятся только пути
  fallback: true; //все возвращенное из функции будет перенесено в html
  fallback: 'blocking'; //

};
```

Пример 2. Динамические параметры

```tsx
function Post({ post }) {
  // Render post...
}

// This function gets called at build time
export async function getStaticPaths() {
  // Call an external API endpoint to get posts
  const res = await fetch("https://.../posts");
  const posts = await res.json();

  // Get the paths we want to pre-render based on posts
  const paths = posts.map((post) => ({
    params: { id: post.id },
  }));

  // We'll pre-render only these paths at build time.
  // { fallback: false } means other routes should 404.
  return { paths, fallback: false };
}

// This also gets called at build time
export async function getStaticProps({ params }) {
  // params contains the post `id`.
  // If the route is like /posts/1, then params.id is 1
  const res = await fetch(`https://.../posts/${params.id}`);
  const post = await res.json();

  // Pass post data to the page via props
  return { props: { post } };
}
```
