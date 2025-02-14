# Utility types

## Partial

Создает тип со всеми свойствами Type, установленными как необязательные. Эта утилита вернет тип, представляющий все подмножества данного типа.

- есть проблема с Partial, если объект формируется динамически, то пропускает поля, которых нет в объекте

```ts
interface Todo {
  title: string;
  description: string;
}
function updateTodo(todo: Todo, fieldsToUpdate: Partial<Todo>) {
  return { ...todo, ...fieldsToUpdate };
}
const todo1 = {
  title: "organize desk",
  description: "clear clutter",
};
const todo2 = updateTodo(todo1, {
  description: "throw out trash",
});
```

## Required

Создает тип, состоящий из всех свойств Type, для которых установлено значение required. Противоположность Partial, все поля интерфейса должны быть обязательными

```ts
interface Props {
  a?: number;
  b?: string;
}
const obj: Props = { a: 5 };
const obj2: Required<Props> = { a: 5 };
// Property 'b' is missing in type '{ a: number; }' but required in type 'Required<Props>'.
```

## Omit, Pick, Extract, Exclude

```ts
interface PaymentPersistent {
  id: number;
  sum: number;
  from: string;
  to: string;
}
// убрать что то из типа убрать id выше
type Payment = Omit<PaymentPersistent, "id">;
/* type Payment = {
    sum: number;
    from: string;
    to: string;
} */
//взять что-то из типов , взять только 'from' и 'to'
type PaymentRequisite = Pick<PaymentPersistent, "from" | "to">;
/* type PaymentRequisite = {
    from: string;
    to: string;
} */
// Забираем только тот тип, который указан в конце
type ExtractEx = Extract<"from" | "to" | Payment, string>;
/* type ExtractEx = "from" | "to" */
//исключить всех, кто относится к типу, указному в generic в конце
type ExcludeEx = Exclude<"from" | "to" | Payment, string>;
/* type ExcludeEx = {
    sum: number;
    from: string;
    to: string;
} */
```

## Return type. Parameters

```ts
//ReturnType - вытаскивает что возвращает функция
class User1 {
  //класс для получения из функции getData() ниже
  constructor(public id: number, public name: string) {}
}
function getData(id: number) {
  //функция по получению нового экземпляра User1
  return new User1(id, "Vasya");
}
//для получения возврата функции
type RT = ReturnType<typeof getData>; //type RT = User1
//для получения параметров
type PT = Parameters<typeof getData>; //type PT = [id: number]
type first = PT[0]; //type first = number
//для получения параметров конструктора
type CP = ConstructorParameters<typeof User1>; //type CP = [id: number, name: string]
```

## Awaited

```ts
//вытаскиваем из разной вложенности
type A = Awaited<Promise<string>>;
type A2 = Awaited<Promise<Promise<string>>>;
interface IMenu {
  name: string;
  url: string;
}
async function getMenu(): Promise<IMenu[]> {
  return [
    { name: "name1", url: "url1" },
    { name: "name2", url: "url2" },
  ];
}
//использование 1 - получить результат из асинхронной функции
type R = Awaited<ReturnType<typeof getMenu>>; //type R = IMenu[]
async function getArray<T>(x: T) {
  return [await x];
}
```
