# Class decorators

Примеры использования - изменение при вызове, мета дата
Принимает два параметра - target (класс), контекст
Контекст:

- kind: 'class'
- name: string
- addInitializer(initializer: (this: class) => void): void
- metadata

```ts
function Component(id: number) {
  //декоратор с дополнительными аргументами (1)
  console.log("init component"); //без явного запуска декоратор запускает функцию
  return (target: Function) => {
    //при доп. аргументах должна возвращать функцию с целевым объектом
    console.log("run component"); //компонент запущен
    target.prototype.id = id; //меняем параметр
  };
}
function logger() {
  //дополнительный декоратор
  console.log("init logger");
  return (target: Function) => {
    console.log("run logger");
  };
}

@logger() //инициализируются сверху вниз но реализация наоборот
@Component(1)
export class User {
  @Prop id: number;
  @Method
  updatedId(@Param newId: number) {
    this.id = newId;
    return this.id;
  }
}
console.log(new User().id);
console.log(new User().updatedId(2));
```
