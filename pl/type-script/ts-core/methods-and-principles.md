## Сужение типов

Из ванильного JS:

- typeof
- Проверка на true
- Строгое сравнение ===
- Оператор in
- Оператор instanceof

Из TS:

- Предсказатель типов is
- ! после свойств, которые возможно undefined
- never
- Контроль типов на лету (let x = math.random() > 0,5 ? string : number)

## not a null assertion

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

## typeGuard

```ts
interface User {
  name: string;
  email: string;
  login: string;
}
interface Admin {
  name: string;
  role: number;
}
const user: User = {
  name: "name",
  email: "email",
  login: "login",
};
//функция для проверки примитивов
function logId(id: string | number) {
  if (isString(id)) {
    //функция для для проверки typeGuard
    console.log(id);
  } else {
    console.log(id);
  }
}
function isString(x: string | number): x is string {
  //приведения
  return typeof x === "string";
}

//typeGuard для Объектов
function isAdmin(user: User | Admin) : user is Admin{user явно приравняло к админу
    return 'role' in user
}
function setRole(user: User | Admin){
    if(isAdmin(user)) {
        user.role = 0;
    } else {
        throw new Error("user isn't admin")
    }
}

```
