# Контекст

Создание контекста для роутинга

```tsx
//_root.tsx
import {
  createRootRouteWithContext,
  createRouter,
} from "@tanstack/react-router";

interface MyRouterContext {
  user: User;
}

// Use the routerContext to create your root route
const rootRoute = createRootRouteWithContext<MyRouterContext>()({
  component: App,
});

//main.tsx
// Use the routerContext to create your router
const router = createRouter({
  routeTree,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootElement = document.getElementById("root")!;
if (!rootElement.innerHTML) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>
  );
}
```

# сброс контекста

```tsx
import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useEffect } from "react";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = () => {
      router.invalidate();
    };

    return unsubscribe;
  }, []);

  return <></>;
}
```

# создание breadcrumbs с помощью контекста

```tsx
export const Route = createRootRoute({
  component: () => {
    const matches = useRouterState({ select: (s) => s.matches });

    const matchWithTitle = [...matches]
      .reverse()
      .find((d) => d.context.getTitle);

    const title = matchWithTitle?.context.getTitle() || "My App";

    return (
      <html>
        <head>
          <title>{title}</title>
        </head>
        <body>{/* ... */}</body>
      </html>
    );
  },
});
```
