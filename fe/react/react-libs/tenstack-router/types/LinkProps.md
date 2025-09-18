# LinkProps

```ts
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
