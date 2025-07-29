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
