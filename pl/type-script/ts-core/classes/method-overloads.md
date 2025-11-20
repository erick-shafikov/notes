# Перегрузка методов

```ts
class User {
  skills: string[];
  addSkill(skill: string): void; //в зависимости от типа аргумента позволяет реализовать разный функционал
  addSkill(skill: string[]): void;
  addSkill(skill: string | string[]): void {
    if (typeof skill === "string") {
      this.skills.push(skill);
    } else {
      this.skills.concat(skill);
    }
  }
} //перегрузка функций
function run(distance: string): string;
function run(distance: number): number;
function run(distance: number | string): number | string {
  if (typeof distance === "number") {
    return 1;
  } else {
    return "";
  }
}
```

# Перегрузка методов с Conditional types

```ts
// перегрузку методов с
class User {
  id: number;
  name: string;
}
class UserPersistent extends User {
  dbId: string;
}
// В случае перегрузки
function getUser(id: number): User;
function getUser(dbId: string): UserPersistent;
function getUser(dbIDorId: string | number): User | UserPersistent {
  if (typeof dbIDorId === "number") {
    return new User();
  } else {
    return new UserPersistent();
  }
}

const res = getUser2(1); //const res: User
const res2 = getUser2("user"); //const res2: UserPersistent

type UserOrUserPersistent<T extends string | number> = T extends number
  ? User
  : UserPersistent;

function getUser2<T extends string | number>(id: T): UserOrUserPersistent<T> {
  if (typeof id === "number") {
    return new User() as UserOrUserPersistent<T>;
  } else {
    return new UserPersistent();
  }
}
const res = getUser2(1); //const res: User
const res2 = getUser2("user"); //const res2: UserPersistent
```
