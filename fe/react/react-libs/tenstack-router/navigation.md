# типы

```ts
type ToOptions<
  TRouteTree extends AnyRoute = AnyRoute,
  TFrom extends RoutePaths<TRouteTree> | string = string,
  TTo extends string = ""
> = {
  from?: string;
  to: string;
  params:
    | Record<string, unknown>
    | ((prevParams: Record<string, unknown>) => Record<string, unknown>);
  search:
    | Record<string, unknown>
    | ((prevSearch: Record<string, unknown>) => Record<string, unknown>);
  hash?: string | ((prevHash: string) => string);
  state?:
    | Record<string, any>
    | ((prevState: Record<string, unknown>) => Record<string, unknown>);
};

//NavigateOptions расширяет ToOptions
export type NavigateOptions<
  TRouteTree extends AnyRoute = AnyRoute,
  TFrom extends RoutePaths<TRouteTree> | string = string,
  TTo extends string = ""
> = ToOptions<TRouteTree, TFrom, TTo> & {
  replace?: boolean;
  resetScroll?: boolean;
  hashScrollIntoView?: boolean | ScrollIntoViewOptions;
  viewTransition?: boolean | ViewTransitionOptions;
  ignoreBlocker?: boolean;
  reloadDocument?: boolean;
  href?: string;
};

//расширяет NavigateOptions
export type LinkOptions<
  TRouteTree extends AnyRoute = AnyRoute,
  TFrom extends RoutePaths<TRouteTree> | string = string,
  TTo extends string = ""
> = NavigateOptions<TRouteTree, TFrom, TTo> & {
  target?: HTMLAnchorElement["target"];
  activeOptions?: {
    exact?: boolean; //false
    includeHash?: boolean; //false
    includeSearch?: boolean;
    explicitUndefined?: boolean;
  };

  preload?: false | "intent"; //предварительная загрузка
  preloadDelay?: number; //задержка при предварительной загрузке
  disabled?: boolean;
};
```

# типы навигации

## Link компонент

```ts
//помимо LinkOptions
export type LinkProps<
  TFrom extends RoutePaths<RegisteredRouter["routeTree"]> | string = string,
  TTo extends string = ""
> = LinkOptions<RegisteredRouter["routeTree"], TFrom, TTo> & {
  activeProps?:
    | FrameworkHTMLAnchorTagAttributes
    | (() => FrameworkHTMLAnchorAttributes);

  inactiveProps?:
    | FrameworkHTMLAnchorAttributes
    | (() => FrameworkHTMLAnchorAttributes);
};
```

пример ссылки с параметрами

```tsx
const link = () => (
  <Link
    to="/blog/post/$postId"
    params={{
      postId: "my-first-blog-post",
    }}
    //удаление параметров
    params={{ category: undefined }}
    params={{}}
    //необязательные
    params={{ category: undefined, slug: undefined }}
    //функционально обновление, императивная навигация
    params={(prev) => ({ ...prev, category: undefined })}
    //
    //параметры поиска
    search={{
      query: "tanstack",
    }}
    // обновить точечно один параметр
    search={(prev) => ({
      ...prev,
      page: prev.page + 1,
    })}
    //к определенному id
    hash="section-1"
  >
    Blog Post
  </Link>
);
```

```tsx
const LinkWithPrefix = () => (
  <Link to="/files/prefix{-$name}.txt" params={{ name: undefined }}>
    Default File
  </Link>
);
```

```tsx
const StyledLink = () => (
  <Link
    to="/blog/post/$postId"
    params={{
      postId: "my-first-blog-post",
    }}
    activeProps={{
      style: {
        fontWeight: "bold",
      },
    }}
    activeOptions={{
      exact: true,
      includeHash: false,
      includeSearch: false,
      explicitUndefined: true,
    }}
  >
    Section 1
  </Link>
);
```

специальные параметры to:

- . - перезагрузка текущего
- .. - назад на один

## Хук useNavigate

```tsx
function Component() {
  const navigate = useNavigate({ from: "/posts/$postId" });

  const handleSubmit = async (e: FrameworkFormEvent) => {
    e.preventDefault();

    if (response.ok) {
      navigate({ to: "/posts/$postId", params: { postId } });
    }
  };
}
```

## Navigate компонент

при необходимости навигации при монтировании компонента

```tsx
function Component() {
  return <Navigate to="/posts/$postId" params={{ postId: "my-first-post" }} />;
}
```

## Router.navigate метод

тоже самое что и две функции выше

# сопоставление пути

```tsx
function Component() {
  const matchRoute = useMatchRoute();

  useEffect(() => {
    if (matchRoute({ to: "/users", pending: true })) {
      console.info("The /users route is matched and pending");
    }
  });

  return (
    <div>
      <Link to="/users">
        Users
        <MatchRoute to="/users" pending>
          <Spinner />
        </MatchRoute>
      </Link>
      {/* или */}
      <Link to="/users">
        Users
        <MatchRoute to="/users" pending>
          {(match) => {
            return <Spinner show={match} />;
          }}
        </MatchRoute>
      </Link>
    </div>
  );
}
```
