# useMatchRoute

Параметры:

- Принимает (объект):
- -
- Возвращает:
- - matchRoute - функция, которая возвращает информацию о текущей локации или загружаемой локации, параметры:
- - - Принимает:
- - - - [UseMatchRouteOptions](../types/UseMatchRouteOptions.md)
- - - Возвращает:
- - - - boolean - конфигурацию роута

---

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

```tsx
import { useMatchRoute } from "@tanstack/react-router";

// Current location: /posts/123
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({ to: "/posts/$postId" });
  //    ^ { postId: '123' }
}

// Current location: /posts/123
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({ to: "/posts" });
  //    ^ false
}

// Current location: /posts/123
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({ to: "/posts", fuzzy: true });
  //    ^ {}
}

// Current location: /posts
// Pending location: /posts/123
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({ to: "/posts/$postId", pending: true });
  //    ^ { postId: '123' }
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({ to: "/posts/$postId/foo/$fooId" });
  //    ^ { postId: '123', fooId: '456' }
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({
    to: "/posts/$postId/foo/$fooId",
    params: { postId: "123" },
  });
  //    ^ { postId: '123', fooId: '456' }
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({
    to: "/posts/$postId/foo/$fooId",
    params: { postId: "789" },
  });
  //    ^ false
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({
    to: "/posts/$postId/foo/$fooId",
    params: { fooId: "456" },
  });
  //    ^ { postId: '123', fooId: '456' }
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({
    to: "/posts/$postId/foo/$fooId",
    params: { postId: "123", fooId: "456" },
  });
  //    ^ { postId: '123', fooId: '456' }
}

// Current location: /posts/123/foo/456
function Component() {
  const matchRoute = useMatchRoute();
  const params = matchRoute({
    to: "/posts/$postId/foo/$fooId",
    params: { postId: "789", fooId: "456" },
  });
  //    ^ false
}
```
