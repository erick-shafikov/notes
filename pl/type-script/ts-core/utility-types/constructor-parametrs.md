```ts
//ReturnType - вытаскивает что возвращает функция
class User1 {
  //класс для получения из функции getData() ниже
  constructor(public id: number, public name: string) {}
}
function getData(id: number) {
  //функция по получению нового экземпляра User1
  return new User1(id, "Vasya");
}
//для получения параметров конструктора
type CP = ConstructorParameters<typeof User1>; //type CP = [id: number, name: string]
```

# Реализация

```ts
type ConstructorParameters<T extends abstract new (...args: any) => any> =
  T extends abstract new (...args: infer P) => any ? P : never;
```
