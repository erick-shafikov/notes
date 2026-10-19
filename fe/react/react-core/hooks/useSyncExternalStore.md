## useSyncExternalStore

useSyncExternalStore(cb, [dep])

```tsx
import { useSyncExternalStore } from "react";
import { todosStore } from "./todoStore.js";

export default function TodosApp(): JSX.Element {
  const todos = useSyncExternalStore(
    todosStore.subscribe,
    todosStore.getSnapshot
  );
  return (
    <>
      <button onClick={() => todosStore.addTodo()}>Add todo</button>
      <hr />
      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>{todo.text}</li>
        ))}
      </ul>
    </>
  );
}
```

```tsx
import { useSyncExternalStore } from "react";

export default function ChatIndicator(): JSX.Element {
  const isOnline = useSyncExternalStore(subscribe, getSnapshot);
  return <h1>{isOnline ? "✅ Online" : "❌ Disconnected"}</h1>;
}

function getSnapshot(): boolean {
  return navigator.onLine;
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
