# provide

позволяет преодолеть проп-дриллинг

- Лучше использовать символы в качестве ключей

```ts
function provide<T>(key: InjectionKey<T> | string, value: T): void;
```

Рекомендуемый подход с реактивными значениями

```vue
<!-- компонент в котором создается контекст -->
<script setup>
import { provide, ref } from "vue";

const location = ref("North Pole");

function updateLocation() {
  location.value = "South Pole";
}

provide("location", {
  location,
  updateLocation,
});
</script>
```

```vue
<!-- компонент который принимает контекст -->
<script setup>
import { inject } from "vue";

const { location, updateLocation } = inject("location");
</script>
```

## readonly

```vue
<script setup>
import { ref, provide, readonly } from "vue";

const count = ref(0);
provide("read-only-count", readonly(count));
</script>
```

## app level

```ts
import { createApp } from "vue";

const app = createApp({});

app.provide(/* key */ "message", /* value */ "hello!");
```

## ts

```ts
import { provide, inject } from "vue";
import type { InjectionKey } from "vue";

const key = Symbol() as InjectionKey<string>;

provide(key, "foo"); // providing non-string value will result in error

const foo = inject(key); // type of foo: string | undefined
```

# inject

```ts
// without default value
function inject<T>(key: InjectionKey<T> | string): T | undefined;

// with default value
function inject<T>(key: InjectionKey<T> | string, defaultValue: T): T;

// with factory
function inject<T>(
  key: InjectionKey<T> | string,
  defaultValue: () => T,
  treatDefaultAsFactory: true
): T;
```

# hasInjectionContext

```ts
function hasInjectionContext(): boolean;
```
