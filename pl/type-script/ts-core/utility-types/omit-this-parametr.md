# Реализация

```ts
type OmitThisParameter<T> = T extends (this: any, ...args: infer A) => infer R
  ? (...args: A) => R
  : T;
```
