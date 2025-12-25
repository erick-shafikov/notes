# стадии

- setup
- beforeCreate
- - здесь происходит инициализация options api
- created
- - здесь определяется есть ли пре-компилированный template, есть нет то собрать шаблон на лету
- beforeMount
- - первый рендер, работаем с dom
- mounted
- - начинается цикл обновлений
- beforeUpdate
- updated
- beforeUnmount
- unmounted

# порядок хуков

```vue
<script setup>
import {
  onMounted,
  ref,
  onUpdated,
  onBeforeMount,
  onBeforeUpdate,
  onBeforeUnmount,
  onUnmounted,
} from "vue";

onBeforeMount(() => {
  console.log("city select before mounted");
});

onMounted(() => {
  console.log("city select mounted");
});

onBeforeUpdate(() => {
  console.log("city select before updated");
});

onUpdated(() => {
  console.log("city select updated");
});

onBeforeUnmount(() => {
  console.log("city select unmount");
});

onUnmounted(() => {
  console.log("city select unmounted");
});
</script>
```

# onMounted()

Вызывает коллбек когда:

- компонент смонтирован
- все его дочерние синхронные компоненты тоже смонтированы
- вставлен в DOM
- нужен для доступа к DOM элемента

```ts
function onMounted(
  callback: () => void,
  target?: ComponentInternalInstance | null
): void;
```

# onUpdated()

вызывается после того как DOM обновился из-за реактивных обновлений, все изменения объединяются в один батч

# onErrorCaptured()

Вызывается коллбек при ошибке рендера компоненте, обработчика события, хука цикла события, setup функции, watch

```ts
function onErrorCaptured(callback: ErrorCapturedHook): void;

type ErrorCapturedHook = (
  err: unknown,
  instance: ComponentPublicInstance | null,
  info: string
) => boolean | void;
```

# dev функции

- onRenderTracked()
- onRenderTriggered()

# keepAlive

- onActivated()
- onDeactivated()

# ssr

- onServerPrefetch()
