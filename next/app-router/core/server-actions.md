# server actions

Асинхронные функции, которые срабатывают на сервере, имеют тег "use server"

- В серверных могут быть созданы с помощью пометки use server в самой первой строке тела функции
- Что бы использовать в клиентских компонентах нужно создать функцию в отдельном файле с пометкой use server и импортировать в клиентский компонент. Могут быть переданы в prop клиентского компонента

Поведение:

- SA могут быть вызваны в action у формы

```tsx
"use server";
export async function serverAction(formData: FromData) {
  return {
    // primitive only
  };
}
```

```tsx
import { serverAction } from "./serverAction.ts";

export const SomeComponent = () => {
  // флаг для отслеживания загрузки
  const { state, fromAction } = useFromStatus(serverAction, {
    //initial state
  });
  return (
    <form action={fromAction}>
      {/* <form action={serverAction}> */}
      <input />
      <input />
      <input />
    </form>
  );
};
```

- форма будет подтверждена, даже если не была загружена на сервере ил не была произведена hydration после hydration форма не перезагрузится
- - отслеживание загрузки useFromStatus
- Могут быть вызваны в обработчиках событий или useEffect
- Валидируются и кэшируются
- Под капотом используют POST метод
- Значения аргументов и возврата должны быть сериализованы React
- Это функции, которые могут быть использованы везде в React
- Используют runtime

Формы. Вызов с аргументами

```js
"use client";
export function UserProfile({ userId }: { userId: string }) {
  const updateUserWithId = updateUser.bind(null, userId);
}
```

```js
"use server";
// ...
export async function updateUser(userId, formData) {
  // ...
}
```

Ревалидирование данных

```js
export async function createPost() {
  try {
    // ...
  } catch (error) {
    // ...
  }
  //ревалидирование по пути, тегу и редирект
  revalidatePath("/posts"); //или revalidateTag('posts') или redirect(`/post/${id}`)
}
```

Можно использовать функции куки для серверных, ревалидирование
Если переменные используются в замыкании, Next автоматически шифрует их
Вызван ы могут быть только если host совпадает с серверным хостом (можно настроить)

```js
module.exports = {
  experimental: {
    serverActions: {
      allowedOrigins: ["my-proxy.com", "*.my-proxy.com"],
    },
  },
};
```
