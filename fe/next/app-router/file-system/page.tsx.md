файл с UI страницы

```tsx
export default function Page({
  params,
  searchParams,
}: {
  //что бы достать params, searchParams нужно использовать sync await или функцию use
  //если params и searchParams используются на клиенте то нужно использовать только с use
  params: Promise<{ slug: string }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}) {
  return <h1>My Page</h1>;
}
```
