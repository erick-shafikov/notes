# cookies

cookies— это асинхронная функция, которая позволяет считывать файлы cookie входящих

```tsx
import { cookies } from "next/headers";

export default async function Page() {
  const cookieStore = await cookies();
  const theme = cookieStore.get("theme");
  return "...";
}
```

# методы

```ts
cookieStore.get("name");
cookieStore.getAll();
cookieStore.has("name");
cookieStore.set(name, value, {
  name,
  value,
  expires, //Date
  maxAge, //number
  domain,
  path,
  secure, // bool
  httpOnly, //bool
  ameSite, //'lax', 'strict','none'
  encode('value'),
  partitioned,
}); //Server Action или Route Handler
cookieStore.clear();
cookieStore.toString();
```
