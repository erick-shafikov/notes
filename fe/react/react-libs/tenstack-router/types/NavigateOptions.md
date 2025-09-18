# NavigateOptions

NavigateOptions расширяет [ToOptions](./ToOptions.md)

```ts
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
