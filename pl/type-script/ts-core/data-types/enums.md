# Enum

Использование

```ts
enum Role {
  Admin,
  Editor,
  Guest,
}

const userRole: Role = Role.Admin;
const userRole: Role = Role.Editor;
//
const userRole: Role = 0; //ок
```

нужен для того, чтобы задать не изменяющийся список, значения которых невозможно переопределить

```ts
ProfessionAction.doctor = "teach";
// Ошибка:
// Cannot assign to 'doctor' because it is a read-only property.ts(2540)

delete ProfessionAction.doctor;
// Ошибка:
// The operand of a 'delete' operator cannot be a read-only property.ts(2704)
```

```ts
type direction = "left" | "right"; //ограниченная структура
enum Direction { // для обозначения ограниченной структуры гетерогенные enum
  Left = 1, //может быть числом
  Right = "right", //гетерогенный enum будет строкой
  Right = "right".length(), //может быть результатом функции, для расчётных enum
}

//использование
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
function objMod(Direction) {} //не будет ошибкой, так как enum ведут себя как объекты
```

использование в качестве ключей объектов и классов

```ts
enum ProfessionAction {
  doctor = "treat",
  teacher = "teach",
}

const professionActions = {
  [ProfessionAction.doctor]: "Лечит пациентов",
  [ProfessionAction.teacher]: "Учит студентов",
};

class ProfessionActions {
  // Используем значения enum как ключи
  [ProfessionAction.doctor]: string;
  [ProfessionAction.teacher]: string;

  constructor() {
    this[ProfessionAction.doctor] = "Лечит пациентов";
    this[ProfessionAction.teacher] = "Учит студентов";
  }
}
```

Перебор значений

```ts
enum ProfessionAction {
  doctor = "treat",
  teacher = "teach",
}

for (let [key, value] of Object.entries(ProfessionAction)) {
  console.log(key, value);
}
```

# Константный enum

```ts
const enum Direction2 { //константный enum в компиляции не будет
  Up,
  Down,
}

let myDirection = Direction2.Up;
```

так как можно переопределить enum

```ts
enum ProfessionAction {
  doctor = "treat",
  teacher = "teach",
}

enum ProfessionAction {
  tailor = "sew",
}
```

```js
"use strict";

var ProfessionAction;

(function (ProfessionAction) {
  ProfessionAction["doctor"] = "treat";
  ProfessionAction["teacher"] = "teach";
})(ProfessionAction || (ProfessionAction = {}));

(function (ProfessionAction) {
  ProfessionAction["tailor"] = "sew";
})(ProfessionAction || (ProfessionAction = {}));
```

что бы исправить

```ts
const enum ProfessionAction {
  doctor = "treat",
  teacher = "teach",
}
// Ошибка:
// Enum declarations can only merge with namespace or other enum declarations.ts(2567)

enum ProfessionAction {
  tailor = "sew",
}
// Ошибка:
// Enum declarations can only merge with namespace or other enum declarations.ts(2567)
```

# автоматические определение полей

```ts
enum StatusCode {
  SUCCESS, //автоматические StatusCode.SUCCESS = 0
  IN_PROCESS, //автоматические StatusCode.IN_PROCESS = 1
  FAILED, //автоматические StatusCode.FAILED =2
}
const res = {
  // значения можно переопределять
  message: "Payment",
  statusCode: StatusCode.SUCCESS, //0
};
```

# минусы

- при компиляции enum, компилятор создает дополнительный JavaScript код и усложняет работу компилятору

```ts
enum ProfessionAction {
  doctor = "treat",
  teacher = "teach",
}
```

преобразуется в

```js
"use strict";

var ProfessionAction;

(function (ProfessionAction) {
  ProfessionAction["doctor"] = "treat";
  ProfessionAction["teacher"] = "teach";
})(ProfessionAction || (ProfessionAction = {}));
```
