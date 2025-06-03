позволяют создавать api-методы с Request и Response

- должны иметь route.js|ts название, не могут быть на одном уровне с layout, page

```ts
//app/api/route.ts
export async function GET(request: Request) {}
```

# кеширование

- кешируются get запросы, по умолчанию кеширование отключено

```ts
// что бы включить
export const dynamic = "force-static";

export async function GET() {
  const res = await fetch("https://data.mongodb-api.com/...", {
    headers: {
      "Content-Type": "application/json",
      "API-Key": process.env.DATA_API_KEY,
    },
  });
  const data = await res.json();

  return Response.json({ data });
}
```

# ре-валидация кеша

```ts
//
export const revalidate = 60;

export async function GET() {
  const data = await fetch("https://api.vercel.app/blog");
  const posts = await data.json();

  return Response.json(posts);
}
```

# куки

!!!TODO связать с cookies api

```ts
//с помощью функции cookies
import { cookies } from "next/headers";

export async function GET(request: Request) {
  const cookieStore = await cookies();
  const token = cookieStore.get("token");

  return new Response("Hello, Next.js!", {
    status: 200,
    headers: { "Set-Cookie": `token=${token.value}` },
  });
}

//или из NextRequest

import { type NextRequest } from "next/server";

export async function GET(request: NextRequest) {
  const token = request.cookies.get("token");
}
```

# заголовки

```ts
//с помощью функции headers
import { headers } from "next/headers";

export async function GET(request: Request) {
  const headersList = await headers();
  const referer = headersList.get("referer");

  return new Response("Hello, Next.js!", {
    status: 200,
    headers: { referer: referer },
  });
}

//с помощью NextRequest
import { type NextRequest } from "next/server";

export async function GET(request: NextRequest) {
  const requestHeaders = new Headers(request.headers);
}
```

## cors

```ts
export async function GET(request: Request) {
  return new Response("Hello, Next.js!", {
    status: 200,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  });
}
```

# redirect

```ts
import { redirect } from "next/navigation";

export async function GET(request: Request) {
  redirect("https://nextjs.org/");
}
```

# динамические параметры строки запроса

```ts
export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> }
) {
  //путь app/items/[slug]/route.js
  //строка запроса /items/a
  //результат Promise<{ slug: 'a' }>
  const slug = (await params).slug; // 'a', 'b', or 'c'
}
```

# query параметры

```ts
import { type NextRequest } from "next/server";

export function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const query = searchParams.get("query");
  // query is "hello" for /api/search?query=hello
}
```

# тело запроса

```ts
export async function POST(request: Request) {
  const res = await request.json();
  return Response.json({ res });
}

//form data
export async function POST(request: Request) {
  const formData = await request.formData();
  const name = formData.get("name");
  const email = formData.get("email");
  return Response.json({ name, email });
}
```

# webhooks

```ts
export async function POST(request: Request) {
  try {
    const text = await request.text();
    // Process the webhook payload
  } catch (error) {
    return new Response(`Webhook error: ${error.message}`, {
      status: 400,
    });
  }

  return new Response("Success!", {
    status: 200,
  });
}
```

## настройки

```ts
export const dynamic = "auto";
export const dynamicParams = true;
export const revalidate = false;
export const fetchCache = "auto";
export const runtime = "nodejs";
export const preferredRegion = "auto";
```
