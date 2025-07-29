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
