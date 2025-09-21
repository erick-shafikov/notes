# Route

# методы

## addChildren

```ts
type TAddChildren = (children: Route[]) => this;
```

добавляет дочерние роуты

## update

```ts
type TUpdate = (options: Partial<UpdatableRouteOptions>) => this;
```

обновление роута

## lazy

```ts
type TLazy = (
  lazyImporter: () => Promise<Partial<UpdatableRouteOptions>>
) => ths;
```

добавляет lazy-роут

# другие методы

[доступны и другие методы](../types/RouteApi.md)
