```tsx
import { connection } from "next/server";

export default async function Page() {
  await connection();
  // не будет включен в пре-рендеринг
  const rand = Math.random();
  return <span>{rand}</span>;
}
```
