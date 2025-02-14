## Enum

```ts
type direction = "left" | "right"; //ограниченная структура
enum Direction { // для обозначения ограниченной структуры гетерогенные enum
  Left = 1, //может быть числом
  Right = "right", //гетерогенный enum будет строкой
  Right = "right".length(), //может быть результатом функции, для расчётных enum
}

function move(direction: Direction) {
  switch (direction) {
    case Direction.Left:
      return -1;
    case Direction.Right:
      return 1;
  }
}
// использование enum как объекта
function objMod(obj: { Left: number }) {}
objMod(Direction); //не будет ошибкой, так как enum ведут себя как объекты
// Константный enum
const enum Direction2 { //константный enum в компиляции не будет
  Up,
  Down,
}

let myDirection = Direction2.Up;

// enum (автоматические оледенение)

enum StatusCode {
  SUCCESS, //автоматические StatusCode.SUCCESS = 0
  IN_PROCESS, //автоматические StatusCode.IN_PROCESS = 1
  FAILED, //автоматические StatusCode.FAILED =2
}
const res = {
  message: "Payment",
  statusCode: StatusCode.SUCCESS, //0
};
```
