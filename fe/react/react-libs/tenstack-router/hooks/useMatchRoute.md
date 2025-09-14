# useMatchRoute

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
