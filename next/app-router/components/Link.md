## Link

Позволяет осуществить переход меду роутами. Основная фишка – пре-фетчинг данных
scroll={false} – если сбросить сохранение позиции при скролле

```tsx
<Link
  //ссылка на ресурс
  href="/dashboard"
  replace={false} //вставить в историю вместо текущей
  scroll={true} //прокрутка
  prefetch={true} //путь и данные будут предзагружены
  legacyBehavior={false} //передавать тег a в качестве дочернего
  passHref={false} //перенаправлять href дочернему компоненту
  shallow={false} //обновить путь пере запуском  getStaticProps, getServerSideProps, getInitialProps
  locale={true}
>
  Dashboard
</Link>
```

# объект в качестве href

```tsx
import Link from "next/link";

function Home() {
  return (
    <Link
      href={{
        pathname: "/blog/[slug]",
        query: { slug: "my-post" },
      }}
    >
      Blog Post
    </Link>
  );
}

export default Home;
```

<!-- BP ---------------------------------------->

# BP

## ссылка

```tsx
import Link from "next/link";
import styled from "styled-components";

// This creates a custom component that wraps an <a> tag
const RedLink = styled.a`
  color: red;
`;

function NavLink({ href, name }) {
  return (
    // для чего нужны passHref legacyBehavior
    <Link href={href} passHref legacyBehavior>
      <RedLink>{name}</RedLink>
    </Link>
  );
}

export default NavLink;
```

с frowardRef

```tsx
import Link from "next/link";
import React from "react";

// Define the props type for MyButton
interface MyButtonProps {
  onClick?: React.MouseEventHandler<HTMLAnchorElement>;
  href?: string;
}

// Use React.ForwardRefRenderFunction to properly type the forwarded ref
const MyButton: React.ForwardRefRenderFunction<
  HTMLAnchorElement,
  MyButtonProps
> = ({ onClick, href }, ref) => {
  return (
    <a href={href} onClick={onClick} ref={ref}>
      Click Me
    </a>
  );
};

// Use React.forwardRef to wrap the component
const ForwardedMyButton = React.forwardRef(MyButton);

export default function Home() {
  return (
    <Link href="/about" passHref legacyBehavior>
      <ForwardedMyButton />
    </Link>
  );
}
```

## редирект в middleware

```ts
import { NextResponse } from "next/server";

export function middleware(request: Request) {
  const nextUrl = request.nextUrl;
  if (nextUrl.pathname === "/dashboard") {
    if (request.cookies.authToken) {
      return NextResponse.rewrite(new URL("/auth/dashboard", request.url));
    } else {
      return NextResponse.rewrite(new URL("/public/dashboard", request.url));
    }
  }
}
```

альтернатива в link

```tsx
"use client";

import Link from "next/link";
import useIsAuthed from "./hooks/useIsAuthed"; // Your auth hook

export default function Home() {
  const isAuthed = useIsAuthed();
  const path = isAuthed ? "/auth/dashboard" : "/public/dashboard";
  return (
    <Link as="/dashboard" href={path}>
      Dashboard
    </Link>
  );
}
```
