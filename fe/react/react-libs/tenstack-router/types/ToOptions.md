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
