/$id - параметры строки

# суффиксы

доступ к суффиксам

```tsx
// src/routes/posts/post-{$postId}.tsx
export const Route = createFileRoute("/posts/post-{$postId}")({
  component: PostComponent,
});

function PostComponent() {
  const { postId } = Route.useParams();
  // postId will be the value after 'post-'
  return <div>Post ID: {postId}</div>;
}
```

```tsx
// src/routes/files/{$fileName}txt
export const Route = createFileRoute("/files/{$fileName}.txt")({
  component: FileComponent,
});

function FileComponent() {
  const { fileName } = Route.useParams();
  // fileName will be the value before 'txt'
  return <div>File Name: {fileName}</div>;
}
```

\_splat

```tsx
// src/routes/on-disk/storage-{$}
export const Route = createFileRoute("/on-disk/storage-{$postId}/$")({
  component: StorageComponent,
});

function StorageComponent() {
  const { _splat } = Route.useParams();
  // _splat, will be value after 'storage-'
  // i.e. my-drive/documents/foo.txt
  return <div>Storage Location: /{_splat}</div>;
}
```

```tsx
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/files/_splat")({
  component: FileViewer,
});
// Путь /files/docs/react/router.md отрендерит Открыт файл по пути: docs/react/router.md
function FileViewer() {
  const { splat } = Route.useParams();
  return <h1>Открыт файл по пути: {splat}</h1>;
}
```

необязательные

```tsx
// src/routes/posts/{-$category}.tsx
export const Route = createFileRoute("/posts/{-$category}")({
  component: PostsComponent,
});

// src/routes/posts/{-$category}/{-$slug}.tsx
export const Route = createFileRoute("/posts/{-$category}/{-$slug}")({
  component: PostComponent,
});

// src/routes/users/$id/{-$tab}.tsx
export const Route = createFileRoute("/users/$id/{-$tab}")({
  component: UserComponent,
});
```

# i18n

```tsx
// Route: /{-$locale}/about
export const Route = createFileRoute("/{-$locale}/about")({
  component: AboutComponent,
});

function AboutComponent() {
  const { locale } = Route.useParams();
  const currentLocale = locale || "en"; // Default to English

  const content = {
    en: { title: "About Us", description: "Learn more about our company." },
    fr: {
      title: "À Propos",
      description: "En savoir plus sur notre entreprise.",
    },
    es: {
      title: "Acerca de",
      description: "Conoce más sobre nuestra empresa.",
    },
  };

  return (
    <div>
      <h1>{content[currentLocale]?.title}</h1>
      <p>{content[currentLocale]?.description}</p>
    </div>
  );
}
```

# маскировка путей
