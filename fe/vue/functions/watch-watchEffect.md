# watch

по умолчанию ленивый, источником может быть ref, getter-функция которая возвращает value, реактивный объект, массивы

```ts
// для одного реактивного источник
function watch<T>(
  source: WatchSource<T>,
  callback: WatchCallback<T>,
  options?: WatchOptions
): {
  // возвращает контроллеры
  (): void;
  pause: () => void; //временно остановит наблюдение
  resume: () => void; //восстановит наблюдение
  stop: () => void; // остановит наблюдение
};

// При отслеживании нескольких
// возврат аналогичен
function watch<T>(
  sources: WatchSource<T>[],
  callback: WatchCallback<T[]>,
  options?: WatchOptions
);

// второй параметр
type WatchCallback<T> = (
  value: T, //новое значение
  oldValue: T, //старое значение
  onCleanup: (cleanupFn: () => void) => void //функция отчистки
) => void;

type WatchSource<T> =
  | Ref<T> // ref
  | (() => T) // getter
  | (T extends object ? T : never); // reactive object

interface WatchOptions extends WatchEffectOptions {
  immediate?: boolean; // false - триггер на создание
  deep?: boolean | number; // false - максимальная глубинна вложенных
  flush?: "pre" | "post" | "sync"; // 'pre'
  // для дебага
  onTrack?: (event: DebuggerEvent) => void;
  onTrigger?: (event: DebuggerEvent) => void;
  once?: boolean; // false (3.4+)
}
```

```vue
<script setup>
import { onWatcherCleanup, ref, watch } from "vue";

// реактивные данные, на которых будет срабатывать эффект
let city = ref("Moscow");

watch(
  // зависимость
  city,
  // два состояния
  (newValue, oldValue) => {
    console.log(city.value);

    //позволяет отчистить
    onWatcherCleanup(() => {
      console.log("clean function");
    });
  },
  {
    //доп настройки
    immediate: true,
    once: true,
  }
);
</script>
```

Пример когда в качестве ресурса геттер

```vue
<script setup>
const state = reactive({ count: 0 });
watch(
  () => state.count,
  (count, prevCount) => {
    /* ... */
  }
);
</script>
```

при массиве зависимостей

```vue
<script setup>
watch([fooRef, barRef], ([foo, bar], [prevFoo, prevBar]) => {
  /* ... */
});
</script>
```

```vue
<script setup>
// если не деструктурировать возврат
const stop = watchEffect(() => {});
stop();

//при деструктуризации
const { stop, pause, resume } = watch(() => {});

pause();
resume();
stop();
</script>
```

# onWatcherCleanup

```vue
<script setup>
import { onWatcherCleanup } from "vue";

watch(id, async (newId) => {
  const { response, cancel } = doAsyncWork(newId);
  onWatcherCleanup(cancel);
  data.value = await response;
});
</script>
```

# watchEffect

запустить эффект при изменении реактивного состояния

```ts
function watchEffect(
  effect: (onCleanup: (cleanupFn: () => void) => void) => void,
  options?: {
    // post - после изменений в dom
    // sync - перед обновлениям
    flush?: "pre" | "post" | "sync";
    onTrack?: (event: DebuggerEvent) => void;
    onTrigger?: (event: DebuggerEvent) => void;
  }
): {
  (): void; //
  pause: () => void;
  resume: () => void;
  stop: () => void;
};
```

```vue
<script setup>
import { ref, watchEffect } from "vue";

// две зависимости
let city = ref("Moscow");
let isEdited = ref(false);

// все реактивные зависимости внутри коллбека будут вызывать коллбек
const unWatch = watchEffect(() => {
  console.log(city.value);
  console.log(isEdited.value);
});

// функция для отписки
unWatch();
</script>
```

Разновидности:

- watchPostEffect() - flush: 'post'
- watchSyncEffect() - flush: 'sync'
