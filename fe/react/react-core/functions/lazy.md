# Lazy

Будет работать только в паре с loader

```tsx
let routes = createRoutesFromElements(
  <Route path="/" element={<Layout />}>
    <Route path="a" lazy={() => import("./a")} />
    <Route path="b" lazy={() => import("./b")} />
  </Route>
);

export async function loader({ request }: { request: Request }) {
  let data = await fetchData(request);
  return json(data);
}
export function Component(): JSX.Element {
  let data = useLoaderData();
  return <div />;
}
```
