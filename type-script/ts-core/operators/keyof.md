# keyOf

```ts
//Тип для массива
type ArrayLike = { [n: number]: unknown };

type A = keyof ArrayLike; //number

type MapLike = { [k: string]: boolean };
type M = keyof MapLike; //number | string - так как ключ [0] превратится в ["0"]

interface IUser {
  name: string;
  age: number;
}
//Достаем ключи
type KeyOfUser = keyof IUser; //KeyOfUser = 'age' | 'number'
const key: KeyOfUser = "age";

//Функция для того что бы достать ключ из объекта
function getValue<T, K extends keyof T>(obj: T, key: K) {
  return obj[key];
}
const user: IUser = {
  name: "UserName",
  age: 30,
};
const userName = getValue(user, "name"); //UserName
```
