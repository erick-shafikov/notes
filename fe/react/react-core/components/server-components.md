Плюсы:

-Ускорение первой загрузка страницы благодаря серверному рендерингу.
-Улучшение SEO — поисковики сразу видят контент.
-Возможность выполнять ресурсоёмкие задачи — тот же парсинг данных — на сервере.

Минусы

- Нужна поддержка фреймворка (например, Next.js), потому что React сам по себе не управляет сервером.
- Не все компоненты можно сделать серверными — те, что используют window или другие браузерные API, останутся клиентскими.
- Усложняет архитектуру, если мы не привыкли к разделению сервер-клиент.

```js
// server-component.js
"use server";
export async function fetchUserData() {
  const res = await fetch("https://api.example.com/user");
  return res.json();
}
// page.js
import { fetchUserData } from "./server-component";
export default async function UserProfile() {
  const user = await fetchUserData();
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.bio}</p>
    </div>
  );
}
```
