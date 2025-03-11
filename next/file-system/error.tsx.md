файл оборачивается в React Error Boundary

```tsx
"use client"; // Error boundaries must be Client Components

import { useEffect } from "react";

//принимает пропс error
export default function Error({
  error,
  reset, //текст ошибки функция для перезагрузки
  message, //текст ошибки
  digest, //хеш сгенерированной ошибки
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error(error);
  }, [error]);

  return (
    <div>
      <h2>Something went wrong!</h2>
      <button
        onClick={
          // Attempt to recover by trying to re-render the segment
          () => reset()
        }
      >
        Try again
      </button>
    </div>
  );
}
```

# global-error

для глобальных ошибок

```tsx
//app/global-error.tsx
"use client"; // Error boundaries must be Client Components

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    // global-error must include html and body tags
    <html>
      <body>
        <h2>Something went wrong!</h2>
        <button onClick={() => reset()}>Try again</button>
      </body>
    </html>
  );
}
```
