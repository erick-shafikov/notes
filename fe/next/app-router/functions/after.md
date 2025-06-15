```tsx
import { after } from "next/server";
// Custom logging function
import { log } from "@/app/utils";

export default function Layout({ children }: { children: React.ReactNode }) {
  after(() => {
    // Execute after the layout is rendered and sent to the user
    log();
  });
  return <>{children}</>;
}
```

```ts
import { after } from "next/server";
import { cookies, headers } from "next/headers";
import { logUserAction } from "@/app/utils";

export async function POST(request: Request) {
  // Perform mutation
  // ...

  // Log user activity for analytics
  after(async () => {
    const userAgent = (await headers().get("user-agent")) || "unknown";
    const sessionCookie =
      (await cookies().get("session-id"))?.value || "anonymous";

    logUserAction({ sessionCookie, userAgent });
  });

  return new Response(JSON.stringify({ status: "success" }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
```
