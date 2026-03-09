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

# RSP

передача между сервером и клиентом происходит с помощь RSP - react server payload. Имеет JSON подобный синтаксис, который характеризует дерево компонентов

```jsx
// серверный компонент
import { Counter } from "./client";

export default function App() {
  return (
    <div>
      <h1>Counter</h1>
      <Counter initialCount={0} />
    </div>
  );
}
```

```json
// который в свою очередь превратится при передачи
{
  "0": [
    "$",
    "div",
    null,
    {
      "children": [
        ["$", "h1", null, { "children": "Counter" }],
        ["$", "$L1", null, { "initialCount": 0 }]
      ]
    }
  ]
}
```

```jsx
// клиентский компонент
import { useState } from "react";

export function Counter({ initialCount }) {
  const [count, setCount] = useState(initialCount);

  return (
    <div>
      <p>Count: {count}</p>
      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={() => setCount((c) => c - 1)}>−</button>
        <button onClick={() => setCount((c) => c + 1)}>+</button>
      </div>
    </div>
  );
}
```

будет иметь вид (этот payload приходит сначала)

```json
{
  1:I["client",[],"Counter"]
}
```

так как он клиентски, react знает что он есть в бандле и подставит его в указанное место с указанными пропсами
