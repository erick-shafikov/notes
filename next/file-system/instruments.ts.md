для отслеживания производительности приложения

```ts
//instruments.ts
import { registerOTel } from "@vercel/otel";

export function register() {
  registerOTel("next-app");
}
```
