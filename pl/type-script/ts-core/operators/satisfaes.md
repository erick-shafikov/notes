# satisfies

это оператор проверки соответствия типа. Проверить, что значение соответствует типу, но не потерять точный вывод типов (inference).Проверит общий тип и по значению определит более узкий тип. satisfies не меняет runtime вообще. Это только проверка типов компилятором

В JS его не существует.

```ts
// Record<...> принудительно расширяет тип
const dataEntries: Record<string | number> = {
  entry1: 1,
  entry2: 2,
};

dataEntries.entry3; //нет ошибки
```

```ts
const dataEntries = {
  entry1: 1,
  entry2: 2,
} satisfies Record<string | number>;

dataEntries.entry3; //ошибка
```

```ts
type Role = "admin" | "user" | "guest";

const roleNames = {
  admin: 1,
  user: 2,
  guest: 3,
} as const satisfies Record<Role, number>;

const r = roleNames.admin;
```
