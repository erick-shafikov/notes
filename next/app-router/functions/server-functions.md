# создание

```js
"use server";

export async function createPost(formData: FormData) {
  const res = await fetch("https://...");
  const json = await res.json();

  //можно вернуть объект
  if (!res.ok) {
    return { message: "Please enter a valid email" };
  }
  //после отработки функции можно:
  //обновить данные
  revalidatePath("/posts");
  //перенаправить
  redirect("/posts");
}
```

могут быть объявлены в серверном компоненте

```tsx
export default function Page() {
  // Server Action
  async function createPost() {
    "use server";
    // Update data
    // ...
  }

  return <></>;
}
```

# вызов

в клиентском компоненте

```tsx
"use client";

import { incrementLike } from "./actions";

export default function LikeButton({ initialLikes }: { initialLikes: number }) {
  const [likes, setLikes] = useState(initialLikes);

  return (
    <>
      <p>Total Likes: {likes}</p>
      <button
        onClick={async () => {
          const updatedLikes = await incrementLike();
          setLikes(updatedLikes);
        }}
      >
        Like
      </button>
    </>
  );
}
```

для работы с формой

```ts
"use server";

export async function createPost(formData: FormData) {
  const title = formData.get("title");
  const content = formData.get("content");

  // Update data
  // Revalidate cache
}
```

# useActionState

```tsx
"use client";

import { useActionState } from "react";
import { createPost } from "@/app/actions";
import { LoadingSpinner } from "@/app/ui/loading-spinner";

export function Button() {
  // так же позволяет отслеживать ошибки
  const [state, action, pending] = useActionState(createPost, false);

  return (
    <button onClick={async () => action()}>
      {pending ? <LoadingSpinner /> : "Create Post"}
    </button>
  );
}
```
