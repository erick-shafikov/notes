# Реализация

```ts
type ThisParameterType<T> = T extends (this: infer U, ...args: any[]) => any
  ? U
  : unknown;
```
