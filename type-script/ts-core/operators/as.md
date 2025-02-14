## Кастомизация типов

```ts
function fetchWithAuth(url: string, method: "post" | "get"): 1 | -1 {
  //может принимать аргумент method только строки post и get
  return 1;
}
fetchWithAuth("s", "post");
let method = "3"; //некоторый переменный метод
fetchWithAuth("s", method as "post"); //кастомизация типов, так как функция может принимать только post и get, с помощью кастомизации можем привести к типу

const myCanvas = document.getElementById("main_canvas") as HTMLCanvasElement; //пример, когда мы знаем, что элемент будет конкретного типа
const myCanvas = <HTMLCanvasElement>document.getElementById("main_canvas"); //второй вариант через generic
```
