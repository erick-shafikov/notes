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

//для получения параметров
type PT = Parameters<typeof getData>; //type PT = [id: number]
type first = PT[0]; //type first = number
```

# реализация

```ts
type Parameters<T extends (...args: any) => any> = T extends (
  ...args: infer P
) => any
  ? P
  : never;
```
