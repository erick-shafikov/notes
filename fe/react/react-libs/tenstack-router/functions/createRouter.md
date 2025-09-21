# createRouter

создает экземпляр роутера

Параметры:

- [принимает](../types/RouteOptions.md)
- [возвращает](../types/Route.md)

```tsx
import { createRouter, RouterProvider } from "@tanstack/react-router";
import { routeTree } from "./routeTree.gen";

const router = createRouter({
  defaultNotFoundComponent: () => {
    return (
      <div>
        <p>Not found!</p>
        <Link to="/">Go home</Link>
      </div>
    );
  },
});

export default function App() {
  return <RouterProvider router={router} />;
}
```
