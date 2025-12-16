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
//для получения возврата функции
type RT = ReturnType<typeof getData>; //type RT = User1
```

# Реализация

```ts
type ReturnType<T extends (...args: any) => any> = T extends (
  ...args: any
) => infer R
  ? R
  : any;
```
