# Return type. Parameters

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
//для получения параметров
type PT = Parameters<typeof getData>; //type PT = [id: number]
type first = PT[0]; //type first = number
//для получения параметров конструктора
type CP = ConstructorParameters<typeof User1>; //type CP = [id: number, name: string]
```
