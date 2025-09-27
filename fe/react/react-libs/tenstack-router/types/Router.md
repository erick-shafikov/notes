# Router

# Свойства

## state

вернет экземпляр [RouterState](./RouterState.md)

# Методы

## update

обновит роутер с новыми опциями

(newOptions: RouterOptions) => void

## subscribe

(eventType: TType, fn: ListenerFn<RouterEvents[TType]>) => (event: RouterEvent) => void

подписка на [RouterEvent](./RouteEvent.md)

## matchRoutes

(pathname: string, locationSearch?: Record<string, any>, opts?: { throwOnError?: boolean; }) => RouteMatch[]

## cancelMatch

(matchId: string) => void

при вызове

## cancelMatches

() => void

## buildLocation

(opts: BuildNextOptions) => ParsedLocation

## commitLocation

```ts
type commitLocation = (
  location: ParsedLocation & {
    replace?: boolean;
    resetScroll?: boolean;
    hashScrollIntoView?: boolean | ScrollIntoViewOptions;
    ignoreBlocker?: boolean;
  }
) => Promise<void>;
```

## navigate

```ts
type navigate = (options: NavigateOptions) => Promise<void>;
```

## invalidate

```ts
function (opts?: {
  filter?: (d: MakeRouteMatchUnion<TRouter>) => boolean;
  sync?: boolean;
  forcePending?: boolean;
}) => Promise<void>;
```

## clearCache

```ts

function (opts?: {filter?: (d: MakeRouteMatchUnion<TRouter>) => boolean}) => void

```

## load

## preloadRoute

## loadRouteChunk

## matchRoute

## dehydrate

## hydrate

## parseSearch

parseSearch: parseSearchWith(JSON.parse),

## stringifySearch

stringifySearch: stringifySearchWith(JSON.stringify),
