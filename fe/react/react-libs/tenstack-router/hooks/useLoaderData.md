# useLoaderData

Возвращает результат функции `loader` указанного маршрута. Тип выводится автоматически из `loader` — вручную писать generic не нужно.

Параметры (объект):

- `from` — путь маршрута, строка, опциональная, но рекомендуется
- `strict` — если `false`, то `from` игнорируется
- `select` — `(loaderData: TLoaderData) => TSelected`, трансформация результата
- `structuralSharing` — boolean для `select`, для оптимизации ре-рендеров

## Два варианта вызова

**Через объект Route** — только для текущего маршрута:

```tsx
export const Route = createFileRoute('/users/$userId')({
  loader: async ({ params }) => {
    return {
      user: await getUser(params.userId),
      permissions: ['read', 'edit'],
    }
  },
})

function UserPage() {
  const data = Route.useLoaderData()
  // data: { user: User; permissions: string[] }
}
```

**Через импортируемый хук** — можно обратиться к loader'у любого маршрута, в том числе родительского:

```tsx
import { useLoaderData } from '@tanstack/react-router'

const data = useLoaderData({ from: '/users/$userId' })
```

## Доступ к данным родительского маршрута

Дочерний маршрут может получить данные loader'а родителя через `from`:

```tsx
// родитель: /dashboard
export const Route = createFileRoute('/dashboard')({
  loader: async () => ({ user: await getCurrentUser() }),
})

// дочерний: /dashboard/settings
function SettingsPage() {
  // берём данные из loader'а родителя
  const { user } = useLoaderData({ from: '/dashboard' })
}
```

`Route.useLoaderData()` без `from` — только данные **своего** loader'а. `useLoaderData({ from })` — данные **любого** маршрута в дереве.

## loader возвращает не только объект

```tsx
loader: async () => await getUsers()
// getUsers(): Promise<User[]>

const users = Route.useLoaderData()
// User[]
```

Схема:

```
loader → Promise<T> | T → useLoaderData() → T
```

## Совместное использование с TanStack Query

`loader` часто используется не для fetch напрямую, а для prefetch через Query:

```tsx
loader: ({ context }) =>
  context.queryClient.ensureQueryData(usersQueryOptions)
```

Тогда `useLoaderData()` вернёт данные из кэша Query, а не сам `QueryClient`.
