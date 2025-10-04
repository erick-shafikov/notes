# Соединение двух объектов

```ts
type Prettify<T> = T extends object ? { [Key in keyof T]: Pretty<T[Key]> } : T;

type Data = Prettify<ShortData & AdditionalData>;
```
