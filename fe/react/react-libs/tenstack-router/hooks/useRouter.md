# preloadRoute

```tsx
function Component() {
  const router = useRouter();

  useEffect(() => {
    async function preload() {
      try {
        const matches = await router.preloadRoute({
          to: postRoute,
          params: { id: 1 },
        });
      } catch (err) {
        // Failed to preload route
      }
    }

    preload();
  }, [router]);

  // несколько

  useEffect(() => {
    async function preloadRouteChunks() {
      try {
        const postsRoute = router.routesByPath["/posts"];
        await Promise.all([
          router.loadRouteChunk(router.routesByPath["/"]),
          router.loadRouteChunk(postsRoute),
          router.loadRouteChunk(postsRoute.parentRoute),
        ]);
      } catch (err) {
        // Failed to preload route chunk
      }
    }

    preloadRouteChunks();
  }, [router]);

  return <></>;
}
```
