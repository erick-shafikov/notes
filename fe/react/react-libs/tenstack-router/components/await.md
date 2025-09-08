# Await

позволяет дождаться отложенных данных

```tsx
// src/routes/posts.$postId.tsx
import { createFileRoute, Await } from "@tanstack/react-router";

export const Route = createFileRoute("/posts/$postId")({
  // ...
  component: PostIdComponent,
});

function PostIdComponent() {
  const { deferredSlowData, fastData } = Route.useLoaderData();

  // do something with fastData

  return (
    <Await promise={deferredSlowData} fallback={<div>Loading...</div>}>
      {(data) => {
        return <div>{data}</div>;
      }}
    </Await>
  );
}
```
