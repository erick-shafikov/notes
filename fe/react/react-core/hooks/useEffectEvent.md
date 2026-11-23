# useEffectEvent

позволяет вытащить нереактивную логику из эффекта

```tsx
const onSomething = useEffectEvent(callback);
```

```tsx
//пример с
import { useEffectEvent, useEffect } from "react";

function ChatRoom({ roomId, theme }) {
  const onConnected = useEffectEvent(() => {
    showNotification("Connected!", theme);
  });

  useEffect(() => {
    const connection = createConnection(serverUrl, roomId);
    connection.on("connected", () => {
      onConnected();
    });
    connection.connect();
    return () => connection.disconnect();
  }, [roomId]);

  // ...
}
```

## Пример

```tsx
// пример без useEffectEvent но с useRef
function MyUserInfo(): JSX.Element {
  const [userName, setUserName] = useState<string>("Bob");
  const nameRef = useRef<string>(userName);
  const [loginMessage, setLoginMessage] = useState<string>("");

  nameRef.current = userName;

  useEffect(() => {
    let loggedInTime = 0;
    const interval = setInterval(() => {
      loggedInTime++;

      setLoginMessage(
        `${nameRef.current} has been logged in for ${loggedInTime} seconds`,
      );
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div>{loginMessage}</div>
      <input
        value={userName}
        onChange={(evt) => setUserName(evt.target.value)}
      />
    </div>
  );
}
```

```tsx
// пример с useEffectEvent
function MyUserInfo(): JSX.Element {
  const [userName, setUserName] = useState<string>("Bob");
  const [loginMessage, setLoginMessage] = useState<string>("");

  const getName = useEffectEvent(() => userName);
  const onTick = useEffectEvent((tick) =>
    setLoginMessage(`${userName} has been logged in for ${tick} seconds`),
  );

  useEffect(() => {
    let ticks = 0;
    const interval = setInterval(() => onTick(++ticks), 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div>{loginMessage}</div>
      <input
        value={userName}
        onChange={(evt) => setUserName(evt.target.value)}
      />
    </div>
  );
}
```

```tsx
//в виде отдельного хука
function useInterval(onTick: (tick: number) => void, timeout: number = 1000) {
  const onTickEvent = useEffectEvent(onTick);
  const getTimeout = useEffectEvent(() => timeout);

  useEffect(() => {
    let ticks = 0;
    let mounted = true;

    function onTick() {
      if (mounted) {
        onTickEvent(++ticks);
        setTimeout(onTick, getTimeout());
      }
    }

    setTimeout(onTick, getTimeout());

    return () => {
      mounted = false;
    };
  }, []);
}
```

## Реализация на useRef (полифил)

Идея: хранить последнюю версию функции в `ref`, а наружу отдавать стабильную обёртку, которая вызывает актуальную функцию через `ref.current`.

```tsx
import { useCallback, useLayoutEffect, useRef } from "react";

function useEventPolyfill<Args extends unknown[], R>(
  callback: (...args: Args) => R,
): (...args: Args) => R {
  const ref = useRef(callback);

  useLayoutEffect(() => {
    ref.current = callback;
  });

  return useCallback((...args: Args) => ref.current(...args), []);
}
```

Почему так:

- **`useLayoutEffect`, а не `ref.current = callback` прямо в рендере** — рендер может быть прерван и отброшен (concurrent-рендеринг), и тогда в `ref` попала бы функция со значениями, которых так и не было на экране. Обновление на коммите (layout effect срабатывает синхронно после коммита, до обычных `useEffect`) гарантирует, что эффекты увидят функцию из закоммиченного рендера. Настоящий React делает так же — подменяет реализацию на коммите.
- **`useLayoutEffect` без массива зависимостей** — `ref` обновляется после каждого рендера.
- **`useCallback(..., [])`** — обёртка имеет стабильную ссылку. Линтер не знает, что результат нашего хука стабилен, и потребует добавить его в зависимости эффекта; т.к. ссылка не меняется, эффект от этого не перезапускается.

Без `useCallback` (минимальный вариант) — вызывать `ref.current()` прямо внутри эффекта:

```tsx
interface ChatRoomProps {
  roomId: string;
}

function ChatRoom({ roomId }: ChatRoomProps): JSX.Element {
  const theme = useContext(ThemeContext);

  const onConnected = (): void => {
    showNotification("Подключено", theme);
  };

  const onConnectedRef = useRef(onConnected);

  useLayoutEffect(() => {
    onConnectedRef.current = onConnected;
  });

  useEffect(() => {
    const connection = createConnection(serverUrl, roomId);
    connection.on("connected", () => onConnectedRef.current());
    connection.connect();
    return () => connection.disconnect();
  }, [roomId]);

  return <h1>Комната {roomId}</h1>;
}
```

Как устроено в самом React (упрощённо): в `memoizedState` хука лежит объект `{ impl }`, наружу возвращается `(...args) => ref.impl(...args)`, а `impl` подменяется на коммите.

### Отличия полифила от настоящего useEffectEvent

- Настоящую функцию можно вызывать **только из эффектов** — при вызове во время рендера React бросает ошибку. Полифил таких проверок не делает.
- Документация не рекомендует передавать функцию из `useEffectEvent` в другие компоненты / хуки как пропс — у полифила то же ограничение по смыслу, хоть он его и не проверяет.
