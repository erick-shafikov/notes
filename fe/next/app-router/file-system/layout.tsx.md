позволяет обернуть страницу и вложенные в нее

- нет searchParams
- общий макет не перерисовывается
- нет доступа к pathname

если нужны searchParams, pathname можно в Layout импортировать клиентский компонент

```tsx
export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;

  params: Promise<{ team: string }>; //динамические параметры маршрута
}) {
  return <section>{children}</section>;
}
```

# RootLayout

должен быть в корне с тегами html и body

```tsx
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```
