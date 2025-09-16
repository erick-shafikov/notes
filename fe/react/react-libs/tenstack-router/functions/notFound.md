```tsx
export const Route = createFileRoute("/posts/$postId")({
  loader: async ({ params: { postId } }) => {
    // Returns `null` if the post doesn't exist
    const post = await getPost(postId);
    if (!post) {
      throw notFound();
      // Alternatively, you can make the notFound function throw:
      // notFound({ throw: true })
    }
    // Post is guaranteed to be defined here because we threw an error
    return { post };
  },
});
```

Проброс из дочернего компонента компонента

```tsx
// _pathlessLayout.tsx
export const Route = createFileRoute("/_pathlessLayout")({
  // This will render
  notFoundComponent: () => {
    return <p>Not found (in _pathlessLayout)</p>;
  },
  component: () => {
    return (
      <div>
        <p>This is a pathless layout route!</p>
        <Outlet />
      </div>
    );
  },
});

// _pathlessLayout/route-a.tsx
export const Route = createFileRoute("/_pathless/route-a")({
  loader: async () => {
    // This will make LayoutRoute handle the not-found error
    throw notFound({ routeId: "/_pathlessLayout" });
    //                      ^^^^^^^^^ This will autocomplete from the registered router
  },
  // This WILL NOT render
  notFoundComponent: () => {
    return <p>Not found (in _pathlessLayout/route-a)</p>;
  },
});
```
