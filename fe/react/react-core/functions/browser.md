# browser (`react-dom`)

`browser()` позволяет компоненту не рендериться на сервере. Вызов `use(browser())` приостанавливает компонент на сервере (показывается Suspense-fallback в HTML), а на клиенте не приостанавливает.

Полезно для компонентов, которым нужны браузерные API недоступные на сервере (timezone, геолокация, `localStorage` и т.п.).

```tsx
import { use } from 'react';
import { browser } from 'react-dom';

function Component(): JSX.Element {
  use(browser());
  // не выполняется на сервере
  return <div />;
}
```

## Пример: часовой пояс

```tsx
import { Suspense, use } from 'react';
import { browser } from 'react-dom';

function TimeZone(): JSX.Element {
  use(browser());
  const timeZone = new Intl.DateTimeFormat().resolvedOptions().timeZone;
  return <p>{timeZone}</p>;
}

export default function App(): JSX.Element {
  return (
    <>
      <p>Your current time zone is:</p>
      <Suspense fallback="Loading...">
        <TimeZone />
      </Suspense>
    </>
  );
}
```

## Условный вызов

`use` можно вызывать после раннего `return` — это единственный хук с таким поведением:

```tsx
interface TimeZoneProps {
  defaultValue?: string;
}

function TimeZone({ defaultValue }: TimeZoneProps): JSX.Element {
  if (defaultValue) {
    return <p>{defaultValue}</p>;
  }

  use(browser()); // вызывается после условного return
  const localTimeZone = new Intl.DateTimeFormat().resolvedOptions().timeZone;
  return <p>{localTimeZone}</p>;
}
```

## В кастомных хуках

```tsx
function useBrowserQuery<T>(query: string, options: { initialData?: T }) {
  if (options.initialData === undefined) {
    use(browser()); // SSR отключается только если нет initialData
  }
  return useQuery(query, options);
}
```
