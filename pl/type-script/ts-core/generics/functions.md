# generic-функции

функция, которая получает разные типы аргументов и возвращает разные типы, при одинаковом функционале

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

## Парные аргументы

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
