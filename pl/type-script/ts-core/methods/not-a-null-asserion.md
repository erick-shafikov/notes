# not a null assertion

```ts
interface Shape {
  kind: "circle" | "square";
  radius?: number; //два необязательных поля
  sideLength?: number;
}

function getArea(shape: Shape) {
  if (shape.kind === "circle") {
    return Math.PI * shape.radius ** 2; //пытаемся что-то сделать с одним из необязательных полей - ошибка //'shape.radius' is possibly 'undefined'.
  }
}
function getArea(shape: Shape) {
  if (shape.kind === "circle") {
    return Math.PI * shape.radius! ** 2; //оператор ! позволят обратится к необязательному полю
  }
}
```
