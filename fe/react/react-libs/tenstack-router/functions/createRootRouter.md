# createRouter

создает файл конфигурации роутера

```tsx
//в main.tsx
const router = createRouter({
  routeTree,
  context: {
    //контекст для роутинга
   },
});

// регистрация типов (по умолчанию)
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
    </StrictMode>,
  );
```
