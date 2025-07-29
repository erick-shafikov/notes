# typeGuard

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
