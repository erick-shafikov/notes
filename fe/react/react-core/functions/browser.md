# browser (`react-dom`)

`browser()` позволяет компоненту не рендериться на сервере. Вызов `use(browser())` приостанавливает компонент на сервере (показывается Suspense-fallback в HTML), а на клиенте не приостанавливает.

Полезно для компонентов, которым нужны браузерные API недоступные на сервере (timezone, геолокация, `localStorage` и т.п.).

```jsx
import { use } from 'react';
import { browser } from 'react-dom';

function Component() {
  use(browser());
  // не выполняется на сервере
}
```

## Пример: часовой пояс

```jsx
import { Suspense, use } from 'react';
import { browser } from 'react-dom';

function TimeZone() {
  use(browser());
  const timeZone = new Intl.DateTimeFormat().resolvedOptions().timeZone;
  return <p>{timeZone}</p>;
}

export default function App() {
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

```jsx
function TimeZone({ defaultValue }) {
  if (defaultValue) {
    return <p>{defaultValue}</p>;
  }

  use(browser()); // вызывается после условного return
  const localTimeZone = new Intl.DateTimeFormat().resolvedOptions().timeZone;
  return <p>{localTimeZone}</p>;
}
```

## В кастомных хуках

```jsx
function useBrowserQuery(query, options) {
  if (options.initialData === undefined) {
    use(browser()); // SSR отключается только если нет initialData
  }
  return useQuery(query, options);
}
```
