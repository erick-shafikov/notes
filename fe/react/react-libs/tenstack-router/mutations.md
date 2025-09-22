# сброс состояния после мутации

```tsx
const router = useRouter();

const addTodo = async (todo: Todo) => {
  try {
    await api.addTodo();
    await router.invalidate({ sync: true });
  } catch {
    //
  }
};

//подписка на мутацию
const router = createRouter();
const coolMutationCache = createCoolMutationCache();

const unsubscribeFn = router.subscribe("onResolved", () => {
  // сброс мутации
  coolMutationCache.clear();
});
```
