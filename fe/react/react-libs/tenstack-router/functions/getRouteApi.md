# getRouteApi

```tsx
// page.tsx
import { getRouteApi } from "@tanstack/react-router"; // доступ к данным [1]

//[1]
const routeApi = getRouteApi("/posts");

function PostComponent() {
  const {
    //получение данных
  } = Route.useLoaderData();

  //[1]
  const data = routeApi.useLoaderData();

  return <></>;
}
```
