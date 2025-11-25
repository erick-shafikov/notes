# Использование generic с interface

```ts
interface ILogLine<T> {
  //в объект кладется в data определенный тип
  timeStamp: Date;
  data: T;
}
const logLine: ILogLine<{ a: number }> = {
  //в данном случае положим {a : 1}
  timeStamp: new Date(),
  data: {
    a: 1,
  },
};
```

# Conditional types

```ts
// Вариант с использованием generic - типами
interface HTTPresponse<T extends "success" | "failed"> {
  code: number;
  data: T extends "success" ? string : Error; //в зависимости от http ответа возвращать соответственные данные
}
const suc: HTTPresponse<"success"> = {
  //использование объект с выполнением вернет этот объект
  code: 200,
  data: "done",
};
const err: HTTPresponse<"failed"> = {
  //обратный случай
  code: 200,
  data: new Error(),
};
```
