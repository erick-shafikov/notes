# LinkOptions

расширяет [NavigateOptions](./NavigateOptions.md)

```ts
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
