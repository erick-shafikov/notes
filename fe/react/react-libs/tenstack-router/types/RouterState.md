# RouterState

```ts
type RouterState = {
  status: "pending" | "idle";
  isLoading: boolean;
  isTransitioning: boolean;
  matches: Array<RouteMatch>;
  pendingMatches: Array<RouteMatch>;
  location: ParsedLocation;
  resolvedLocation: ParsedLocation;
};
```
