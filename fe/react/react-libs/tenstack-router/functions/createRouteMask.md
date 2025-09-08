Позволит замаскировать путь

```tsx
import { createRouteMask } from "@tanstack/react-router";

const photoModalToPhotoMask = createRouteMask({
  routeTree,
  from: "/photos/$photoId/modal",
  to: "/photos/$photoId",
  params: (prev) => ({
    photoId: prev.photoId,
  }),
});

const router = createRouter({
  routeTree,
  routeMasks: [photoModalToPhotoMask],
});
```
