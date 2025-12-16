# Awaited

```ts
//вытаскиваем из разной вложенности
type A = Awaited<Promise<string>>;
type A2 = Awaited<Promise<Promise<string>>>;
interface IMenu {
  name: string;
  url: string;
}
async function getMenu(): Promise<IMenu[]> {
  return [
    { name: "name1", url: "url1" },
    { name: "name2", url: "url2" },
  ];
}
//использование 1 - получить результат из асинхронной функции
type R = Awaited<ReturnType<typeof getMenu>>; //type R = IMenu[]
async function getArray<T>(x: T) {
  return [await x];
}
```

# Реализация

```ts
type Awaited<T> = T extends PromiseLike<infer U> ? Awaited<U> : T;
```
