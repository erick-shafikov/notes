# getServerSideProps (на сервере)

Функция предназначенная для отображения страниц, на которой очень часто меняются данные. используется как серверная функция,

!!! промежуточные данные не попадут на клиент,
!!! вызывается при каждом запросе
!!! Импорты не будут вынесены в клиент

```tsx
// Типы для пропсов
import type { InferGetServerSidePropsType, GetServerSideProps } from "next";

type Repo = {
  name: string;
  stargazers_count: number;
};

// принимает параметр контекста
export const getServerSideProps = (async (ctx: GetServerSidePropsContext) => {
  const res = await fetch("https://api.github.com/repos/vercel/next.js");
  const repo: Repo = await res.json();
  return { props: { repo } };
}) satisfies GetServerSideProps<{ repo: Repo }>;

// в клиентском компоненте принимает в качестве props
export default function Page({
  repo,
}: InferGetServerSidePropsType<typeof getServerSideProps>) {
  return (
    <main>
      <p>{repo.stargazers_count}</p>
    </main>
  );
}
```

Типизация параметра контекста

```ts
export type GetServerSidePropsContext<
  Q extends ParsedUrlQuery = ParsedUrlQuery,
  D extends PreviewData = PreviewData
> = {
  // входящий запрос
  req: IncomingMessage & {
    cookies: NextApiRequestCookies;
  };
  // ответ
  res: ServerResponse;
  params?: Q;
  query: ParsedUrlQuery;
  preview?: boolean;
  previewData?: D;
  resolvedUrl: string;
  locale?: string;
  locales?: string[];
  defaultLocale?: string;
};
```

Возвращает значение вида

```tsx
type GetServerSidePropsReturnType = {
  props: Object;
  notFound: boolean; //если вернуть true, то перенаправит на 404
  redirect: {
    destination: string; //куда
    permanent: boolean; //редирект с 308 кодом (true) если нужно закешировать или 307 (false) кодом без кеша
    statusCode: string; //с каким кодом вместо permanent
  };
};
```
