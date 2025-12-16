# Partial

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

# реализация

```ts
type Partial<T> = {
  [K in keyof T]?: T[K];
};
```
