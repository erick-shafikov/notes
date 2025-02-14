## Distributive conditional types

```ts
type ToArray<Type> = Type extends any ? Type[] : never;
// в первом случае все зависит от первого присвоения
type StrArrOrNumArr1 = ToArray<string | number>; //type StrArrOrNumArr1 = string[] | number[]
const strArrOrNumArr1: StrArrOrNumArr1 = ["s"];
const strArrOrNumArr2: StrArrOrNumArr1 = [1];
const strArrOrNumArr3: StrArrOrNumArr1 = ["s", 1]; //ошибка

// при Distributive  разбивается
type ToArrayNonDist<Type> = [Type] extends [any] ? Type[] : never;
// 'StrArrOrNumArr' is no longer a union.
type StrArrOrNumArr = ToArrayNonDist<string | number>; //type StrArrOrNumArr = (string | number)[]
const strArrOrNumArr4: StrArrOrNumArr = ["s", 1]; //ошибки нет

type TToArray<T> = T[]; //тоже самое
```

### Использование generic с interface

```ts
interface ILogLine<T> {
  //в объект кладется в data определенный тип
  timeStamp: Date;
  data: T;
}
const logLine: ILogLine<{ a: number }> = {
  //в данном случае положим {a : 1}
  timeStamp: new Date(),
  data: {
    a: 1,
  },
};
```

### Парные аргументы

```ts
function log<T, K>(obj: T, arr: K[]): K[] {
  //generic который принимает 2 типа аргументов возвращает определённый
  //obj.length() не будет работать так как мы не знаем что будет преданно
  console.log(obj);
  return arr;
}
log<string, number>("sdf", [1]); //применение

// исправим с помощью interface

interface HasLength {
  length: number;
}
function log<T extends HasLength, K>(obj: T, arr: K[]): K[] {
  //первым аргументом будет объект, у которого будет свойство length к которому можно обратиться
  obj.length; //теперь можно
  console.log(obj);
  return arr;
}
// описание методов
interface IUser {
  name: string;
  age: number;
  bid: <T>(sum: T) => boolean;
}
function bid<T>(sum: T): boolean {
  return true;
}
```

## Расширяемый generic

```ts
//c помощью ключевого слова extends мы указываем TS, что generic Type содержит поле length
function longest<Type extends { length: number }>(a: Type, b: Type) {
  if (a.length >= b.length) {
    return a;
  } else {
    return b;
  }
} // longerArray is of type 'number[]'
const longerArray = longest([1, 2], [1, 2, 3]);
// longerString is of type 'alice' | 'bob'
const longerString = longest("alice", "bob");
// Error! Numbers don't have a 'length' property
const notOK = longest(10, 100);
// Argument of type 'number' is not assignable to parameter of type '{ length: number; }'.
```

```ts
class Vehicle {
  //объект
  run!: number;
}

function kmToMiles<T extends Vehicle>(vehicle: T): T {
  //без расширения не определит тип
  vehicle.run = vehical.run / 0.62;
  return vehicle;
}

class LCV extends Vehicle {
  capacity!: number;
}

const vehicle = kmToMiles(new Vehicle());
const lvc = kmToMiles(new LCV());

kmToMiles({ run: 1 }); // тоже сработает так как интерфейс схож

function logId<T extends string | number>(id: T): T {
  console.log(id);
  return id;
}
```

## generic в типизации объектов и функций

```ts
function identity<Type>(arg: Type): Type {
  return arg;
}
// типизируем функцию с дженериком как сигнатуру функции
let myIdentity1: <Type>(arg: Type) => Type = identity;
function identity2<Type>(arg: Type): Type {
  return arg;
}
// типизируем функцию с дженериком как объект
let myIdentity2: { <Type>(arg: Type): Type } = identity; // типизируем функцию с помощью интерфейса
interface GenericIdentityFn {
  <Type>(arg: Type): Type;
}
let myIdentity3: GenericIdentityFn = identity; // типизируем функцию с помощью интерфейса c дженериком
interface GenericIdentityFnWithGen<Type> {
  (arg: Type): Type;
}
let myIdentity4: GenericIdentityFnWithGen<number> = identity;
```

## generic-функции

функция, которая получает разные типы аргументов и возвращает разные типы, при одинаковом функционале

```ts
// проблема
function log(obj: string): string | number {
  //функция которая получает объект, а вернуть может строку или число - плохо
  console.log(obj);
  return obj;
}
function log1(obj: number): number {
  //разбиваем на функцию которая получает строку
  console.log(obj);
  return obj;
}
function log2(obj: string): string {
  //и функция которая получает число нарушаем DRY
  console.log(obj);
  return obj;
}

// решение дупликации кода – generic

function log<T>(obj: T): T {
  //для generic - если приходит тип <T>, то и возвращаем <T> можно использовать любую переменную
  console.log(obj);
  return obj;
}

//использование, строго привязываем тип выполнения
const res1 = log<string>(10); //const res: 10 строго привязал
const res2 = log<number>(10); //const res2: number
log<string>("asd"); //в случае строки
log<number>(4); //в случае числа

//пример 2
function getSpitedHalf<T>(data: Array<T>): Array<T> {
  //если указать просто (data: T): T то обращение к свойству length вызывает ошибку, так как не у всех есть свойство length
  const l = data.length / 2;
  return data.slice(0, l);
}
getSpitedHalf([1, 2, 4]); //function getSpitedHalf<number>(data: number[]): number[] определил сам

const split: <T>(data: Array<T>) => Array<T> = getSpitedHalf; //- передача по ссылке
```

Использование стрелочных функций и generic

```ts
const prepareDataItems = <T>(items: T[]) => {};
```

## Conditional types

```ts
// Вариант с использованием generic - типами
interface HTTPresponse<T extends "success" | "failed"> {
  code: number;
  data: T extends "success" ? string : Error; //в зависимости от http ответа возвращать соответственные данные
}
const suc: HTTPresponse<"success"> = {
  //использование объект с выполнением вернет этот объект
  code: 200,
  data: "done",
};
const err: HTTPresponse<"failed"> = {
  //обратный случай
  code: 200,
  data: new Error(),
};
```

<!-- BP ------------------------------------------------------------------------------------------------------------->

# BP

## пример типизируемого запроса

```ts
export default {};
type TMethods = "GET" | "POST" | "PUT" | "PATCH";
type TOptions = {
  method?: TMethods;
  body?: BodyInit;
};
type TProduct = [{ name: string }];
type TSuccess<T> = {
  res: true;
  data: T;
};
type TError = {
  res: false;
  error: Error;
};
type TResponse<T> = TSuccess<T> | Terror<T>;

async function getJson<S>(
  url: string,
  options: TOptions = {}
): Promise<TResponse<S>> {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    return { res: true, data };
  } catch (e) {
    return { res: false, error: e instanceof Error ? e : new Error("error") };
  }
}
const a = getJson<TProduct>("www").then((res) => {
  if (res.res) {
    console.log(res);
  } else {
    res.error;
  }
});
```
