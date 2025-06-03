# примитивы

```ts
// три базовых типа
let a: number = 5;
let b: string = "4";
let c: boolean = false;

let names: string[] = ["asd", "asd"]; //пример массива строк
let args: number[] = [1, 4]; //пример массива строк
let books: object[] = [
  { name: "Алиса в Стране чудес", author: "Льюис Кэрол" },
  { name: "Идиот", author: "Федор Достоевский" },
]; // может содержать только объекты

// tuple
let tup: [number, string] = [2, "sdf"]; //пример tuple, кортеж - смешанный массив, который будет с фиксированным размером, фиксированными элементами

let e: any = 3; //тип any без привязки к типу, может вызывать ошибки
e = "sdf";
e = true;

let anyArr: any[] = ["someString", true]; //массив со смешенным типом переменных
// Функции
function greet(name: string): string {
  //типизируем аргументы и возвращаемое значение
  return name + "hi";
}
// Для анонимных функций
names.map((x: string) => x); //для анонимных функций
// Объектные типы
function coord(coord: { lat: number; long?: number }) {} //знак вопроса, которое говорит, что number или задан или нет
```

# void

Если функция возвращает хоть в одном случае, то будет возвращаемым значением – undefined
Если функция ничего не возвращает – void

```ts
type VoidFunc = () => void;
const f: VoidFunc = () => {
  //возвращает равно void
  return true;
};

function invokeInFourSeconds(callback: () => undefined) {
  setTimeout(callback, 4000);
}
function invokeInFiveSeconds(callback: () => void) {
  setTimeout(callback, 5000);
}
const values: number[] = [];
invokeInFourSeconds(() => values.push(4)); //ошибка так как push -> длину нового массива
invokeInFiveSeconds(() => values.push(4)); //ошибка нет, так void игнорирует значение

function foo(): void {
  return 42; //ошибка
}
```

# unknown

```ts
// unknown
// переопределение unknown
let input: unknown;
input = 3; //ok
input = ["sd", "sd"]; //ok
let res: string = input; //не ok так как нельзя переопределить unknown Type 'unknown' is not assignable to type 'string'
function run(i: unknown) {
  if (typeof i == "number") {
    i++;
  } else {
    i; //(parameter) i: unknown
  }
}

run(input);

// ошибка – тип unknown

async function getData() {
  try {
    await fetch("");
  } catch (error) {
    console.log(error.message); // не ок 'error' is of type 'unknown'
    if (error instanceof Error) {
      console.log(error.message); // ок так как идет приведение к типу
    }
  }
}
```

разница между any и unknown.

unknown требует проверку типов, any нет

### never

```ts
function generateError(message: string): never {
  //в случае вылета
  throw new Error(message);
}
function dumpError(): never {
  //в случае бесконечного цикла
  while (true) {}
}
function rec(): never {
  //в случае рекурсии
  return rec();
}

const a: never = undefined; //нельзя присвоить ни одного из типа переменных
type paymentAction = "refund" | "checkout";

function processAction(action: paymentAction) {
  switch (action) {
    case "refund": //...
      break;
    case "checkout": //...
      break;
    default:
      const _: never = action; //если добавить в paymentAction еще тип, то вылезет ошибка для переменной _
      throw new Error("no type for these case");
  }
}
```

# union типы

```ts
let universalId: number | string = 5; //universalId может быть и строкой или числом
universalId = "someString";

function printId(id: number | string) {
  //id.toUpperCase() в этом месте будет ошибка, так как тип не определен
  if (typeof id == "string") {
    console.log(id.toUpperCase());
  } else {
    console.log(id);
  }
}

function helloUser(user: string | string[]) {
  // возможны два вариант или строка или массив строк
  if (Array.isArray(user)) {
    //для случая массива можно использовать функционал массивов
    console.log(user.join(", " + "hi"));
  } else {
    console.log(user + "hi"); //в случае строки
  }
}
```

## Литеральные типы

```ts
// пример с примитивами
const a = "someString"; //такую переменную нельзя переопределить, ее тип будет "someString";
let b: "hi" = "hi"; //создает тип Hi

// пример с union
type Direction = "left" | "right";

function moveDog(direction: Direction): -1 | 0 | 1 {
  switch (direction) {
    case "left":
      return -1;
    case "right":
      return 1;
    default:
      return 0;
  }
}

// пример
const connection = {
  host: "localhost",
  protocol: "https" as "https", //строка может только 'https" так как при передачи в функцию будет ожидаться строка
};

const c: any = 5;
//let d = c as number;
let d = <number>c; //небезопасно пример React, так как распознает как jsx элемент
function connect(host: string, protocol: "http" | "https") {}
connect(connection.host, connection.protocol);
```
