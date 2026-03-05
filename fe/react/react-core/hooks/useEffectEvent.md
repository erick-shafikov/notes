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

```jsx
// пример без useEffectEvent но с useRef
function MyUserInfo() {
  const nameRef = useRef(userName);
  const [userName, setUserName] = useState("Bob");
  const [loginMessage, setLoginMessage] = useState("");

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

```jsx
// пример с useEffectEvent
function MyUserInfo() {
  const [userName, setUserName] = useState("Bob");
  const [loginMessage, setLoginMessage] = useState("");

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
