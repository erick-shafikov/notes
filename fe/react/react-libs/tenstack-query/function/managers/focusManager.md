# focusManager

```ts
focusManager.setEventListener((handleFocus) => {
  // Listen to visibilitychange
  if (typeof window !== "undefined" && window.addEventListener) {
    window.addEventListener("visibilitychange", handleFocus, false);
  }

  return () => {
    // Be sure to unsubscribe if a new handler is set
    window.removeEventListener("visibilitychange", handleFocus);
  };
});
```

Методы:

- subscribe

```ts
import { focusManager } from "@tanstack/react-query";

const unsubscribe = focusManager.subscribe((isVisible) => {
  console.log("isVisible", isVisible);
});
```

- setFocused

```ts
import { focusManager } from "@tanstack/react-query";

// Set focused
focusManager.setFocused(true);

// Set unfocused
focusManager.setFocused(false);

// Fallback to the default focus check
focusManager.setFocused(undefined);
```

- isFocused

```ts
const isFocused = focusManager.isFocused();
```
