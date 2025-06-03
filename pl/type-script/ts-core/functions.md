# Функции

Типизация функций

```ts
interface TwoNumberCalculation {
  (x: number, y: number): number; //с помощью интерфейса, так же можно type TwoNumberCalculation = {}
}

type TwoNumberCalc = (x: number, y: number) => number; //с помощью типа
const add: TwoNumberCalculation = (a, b) => a + b;
const subtract: TwoNumberCalc = (a, b) => a - b;

// Типизация параметров
type DescribableFunction = {
  //функцию можно определить как объект, так как у функции могут быть свойства
  description: string;
  (someArg: number): boolean;
};
function doSomething(fn: DescribableFunction) {
  //определяем тип параметра как объект-функции
  console.log(fn.description + " returned " + fn(6)); //вызываем и свойство функции и взываем саму функцию
}

// определяем
function myFunc(someArg: number) {
  return someArg > 3;
}
// добавляем свойство
myFunc.description = "default description";
doSomething(myFunc);
```

## Опциональные параметры

```ts
function f(x?: number) {
  // ...
}
f(); // OK
f(10); // OK
```

## Перегрузка методов с Conditional types

```ts
// перегрузку методов с
class User {
  id: number;
  name: string;
}
class UserPersistent extends User {
  dbId: string;
}
// В случае перегрузки
function getUser(id: number): User;
function getUser(dbId: string): UserPersistent;
function getUser(dbIDorId: string | number): User | UserPersistent {
  if (typeof dbIDorId === "number") {
    return new User();
  } else {
    return new UserPersistent();
  }
}

const res = getUser2(1); //const res: User
const res2 = getUser2("user"); //const res2: UserPersistent

type UserOrUserPersistent<T extends string | number> = T extends number
  ? User
  : UserPersistent;

function getUser2<T extends string | number>(id: T): UserOrUserPersistent<T> {
  if (typeof id === "number") {
    return new User() as UserOrUserPersistent<T>;
  } else {
    return new UserPersistent();
  }
}
const res = getUser2(1); //const res: User
const res2 = getUser2("user"); //const res2: UserPersistent
```
