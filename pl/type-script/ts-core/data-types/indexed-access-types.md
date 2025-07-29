# Indexed Access Types

```ts
interface Role {
  name: string;
}

interface Permission {
  endDate: Date;
}

interface User {
  name: string;
  roles: Role[]; //string[]
  permission: Permission; // {endDate: Date}
}

const user: User = {
  name: "Vasya",
  roles: [],
  permission: {
    endDate: new Date(),
  },
};

// ----------------------------------------------------------------------
const nameUser = user["name"]; //string
let rolesName: "roles" = "roles";
type rolesType = User["roles"]; //type rolesType = Role[]
type rolesType2 = User[typeof rolesName]; //rolesName === 'roles' => rolesType2 === Roles[]
type roleType = User["roles"][number]; //получить элемент массива, ключ === number
type roleType2 = User["permission"]["endDate"]; //получить элемент массива, вытаскиваем из массива тип roleType2 = Date
const roles = ["admin", "user", "super-user"] as const; //теперь это tuple с тремя элементами
type roleTypes = (typeof roles)[number]; //type roleTypes 'admin' | 'user' | 'super-user'

// ----------------------------------------------------------------------
type Person = { age: number; name: string; alive: boolean };
type I1 = Person["age" | "name"]; //string | number определение аллиаса ключей
type AliveOrName = "alive" | "name"; //или так
type I5 = Person[AliveOrName]; //string | boolean

type I3 = Person[keyof Person]; //string | number | boolean использование с keyof
```
