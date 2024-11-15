# Lazy

Будет работать только в паре с loader

```jsx
let routes = createRoutesFromElements(
  <Route path="/" element={<Layout />}>
    <Route path="a" lazy={() => import("./a")} />
    <Route path="b" lazy={() => import("./b")} />
  </Route>
);

export async function loader({ request }) {
  let data = await fetchData(request);
  return json(data);
}
export function Component() {
  let data = useLoaderData();
}
```
