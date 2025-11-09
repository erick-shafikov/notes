# types distribution

раскрытие union-ов - поведение называется дистрибуцией или распределением типов

```ts
type ToArray1<T> = T[];
// ToArray1<string | number> => (string | number)[]

type ToArray2<T> = T extends any ? T[] : never;
// ToArray2<string | number> => string[] | number[]
```

Использование

```ts
type Filter =
  | { type: "text"; value: string }
  | { type: "date"; value: Date }
  | { type: "tags"; value: string[] };

type TupleValues<T extends readonly Filter[]> = {
  [K in keyof T]: T[K] extends Filter // Для каждого индекса кортежа // Если его значение соответствует фильтру
    ? T[K]["value"]
    : never;
};

// Эти фильтры понадобятся нам для построения таблицы или поиска и тп
const filters = [
  { type: "text", value: "foo" },
  { type: "date", value: new Date() },
] as const;

const values: TupleValues<typeof filters> = ["foo", new Date()];
```

```ts
type Events =
  | { type: "user_created"; user: User }
  | { type: "user_deleted"; userId: number }
  | { type: "user_updated"; user: User };

type EventHandler<E> = (event: E) => void;

type EventMap<Ev extends { type: string }> = {
  [K in Ev as K['type']]: EventHandler<Extract<Ev, { type: K['type'] }>> };

type UserEventHandlers = EventMap<Events>; /* {   user_created: (event: { type: 'user_created', user: User }) => void;   user_deleted: (event: { type: 'user_deleted', userId: number }) => void;   user_updated: (event: { type: 'user_updated', user: User }) => void; } /
```

```ts
type Field =
  | { name: "email"; value: string }
  | { name: "age"; value: number }
  | { name: "newsletter"; value: boolean };

type ValidationResult<T> = T extends { value: infer V }
  ? V extends string
    ? { valid: boolean; normalized: string }
    : { valid: boolean }
  : never;

type Results = ValidationResult<Field>;
// -> { valid: boolean; normalized: string } | { valid: boolean }
```
