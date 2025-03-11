Динамическую загрузку компонентов можно обеспечить

# next/dynamic

```tsx
"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

// Client Components:
const ComponentA = dynamic(() => import("../components/A"));
const ComponentB = dynamic(() => import("../components/B"));
//отключит рендер на стороне сервера
const ComponentC = dynamic(() => import("../components/C"), {
  ssr: false,
  loading: () => <p>Loading...</p>,
});
//именованный экспорт
const ClientComponent = dynamic(() =>
  import("../components/hello").then((mod) => mod.Hello)
);

export default function ClientComponentExample() {
  const [showMore, setShowMore] = useState(false);

  const handlerClick = () => setShowMore(!showMore);

  return (
    <div>
      {/* Load immediately, but in a separate client bundle */}
      <ComponentA />

      {/* Load on demand, only when/if the condition is met */}
      {showMore && <ComponentB />}
      <button onClick={handlerClick}>Toggle</button>

      {/* Load only on the client side */}
      <ComponentC />
    </div>
  );
}
```

возможно загрузить и библиотеки

# lazy loading

```js
//объявление в качестве переменной
const LazyComponent = dynamic(() => import("../components/Lazy/Lazy"), {
  ssr: false,
  loading: () => <>Loading...</>,
});
```
