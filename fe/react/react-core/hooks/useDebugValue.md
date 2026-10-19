## useDebugValue

useDebugValue(value, format?) – value будет отображено в консоли разработчика

```tsx
import { useSyncExternalStore, useDebugValue } from "react";

export function useOnlineStatus(): boolean {
  const isOnline = useSyncExternalStore(
    subscribe,
    () => navigator.onLine,
    () => true
  );
  useDebugValue(isOnline ? "Online" : "Offline");
  return isOnline;
}

function subscribe(callback: () => void): () => void {
  window.addEventListener("online", callback);
  window.addEventListener("offline", callback);
  return () => {
    window.removeEventListener("online", callback);
    window.removeEventListener("offline", callback);
  };
}
```
