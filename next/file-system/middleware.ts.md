!!!TODO
используется для исполнения на сервере перед запросом

```ts
import { NextResponse, NextRequest } from "next/server";

export const config = {
  matcher: [
    {
      source: "/api/*",
      regexp: "^/api/(.*)",
      locale: false,
      has: [
        { type: "header", key: "Authorization", value: "Bearer Token" },
        { type: "query", key: "userId", value: "123" },
      ],
      missing: [{ type: "cookie", key: "session", value: "active" }],
    },
  ],
};

// принимает один параметр request
export function middleware(request: NextRequest) {
  return NextResponse.redirect(new URL("/home", request.url));
}

export const config = {
  matcher: "/about/:path*",
};
```
