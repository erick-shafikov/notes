# redirect

используется для перенаправления

```ts
import { redirect } from "@tanstack/react-router";

const route = createRoute({
  // throwing an internal redirect object using 'to' property
  loader: () => {
    if (!user) {
      throw redirect({
        to: "/login",
      });
    }
  },
  // throwing an external redirect object using 'href' property
  loader: () => {
    if (needsExternalAuth) {
      throw redirect({
        href: "https://authprovider.com/login",
      });
    }
  },
  // or forcing `redirect` to throw itself
  loader: () => {
    if (!user) {
      redirect({
        to: "/login",
        throw: true,
      });
    }
  },
  // ... other route options
});
```
