# ToOptions

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
```

# NavigateOptions

```ts
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
```

# LinkOptions

```ts
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

# LinkProps

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

# ValidateLinkOptions

типизация пропсов с помощью ValidateLinkOptions

Есть так же:

- ValidateLinkOptionsArray
- ValidateRedirectOptions
- ValidateNavigateOptions

```tsx
export interface HeaderLinkProps<
  TRouter extends RegisteredRouter = RegisteredRouter,
  TOptions = unknown
> {
  title: string;
  linkOptions: ValidateLinkOptions<TRouter, TOptions>;
}

export function HeadingLink<TRouter extends RegisteredRouter, TOptions>(
  props: HeaderLinkProps<TRouter, TOptions>
): React.ReactNode;
// перегрузка
export function HeadingLink(props: HeaderLinkProps): React.ReactNode {
  return (
    <>
      <h1>{props.title}</h1>
      <Link {...props.linkOptions} />
    </>
  );
}
```
