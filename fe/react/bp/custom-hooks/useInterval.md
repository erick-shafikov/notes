```ts
import { useEffect, useRef, useState } from "react";

export const useInterval = (callback: () => void, interval = 1000) => {
  const [active, setActive] = useState(true);
  const intervalIdRef = useRef<ReturnType<typeof setInterval>>();
  const callbackRef = useRef(callback);

  // Обновляем референс при изменении callback
  callbackRef.current = callback;

  useEffect(() => {
    if (!active) return;

    // Используем callbackRef для доступа к актуальной версии callback
    intervalIdRef.current = setInterval(() => callbackRef.current(), interval);

    // Очистка при размонтировании или изменении зависимостей
    return () => {
      clearInterval(intervalIdRef.current);
    };
  }, [active, interval]);

  return {
    active,
    pause: () => setActive(false),
    resume: () => setActive(true),
    toggle: () => setActive((prev) => !prev),
  };
};
```
