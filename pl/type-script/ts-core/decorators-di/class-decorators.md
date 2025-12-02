# Class decorators

Примеры использования - изменение при вызове, мета дата, менять класс на лету. Должен возвращать новый класс

Принимает два параметра - target (класс) и контекст

```ts
type ClassDecoratorContext = {
  kind: "class";
  name: string; //имя класса
  addInitializer(initializer: (this: class) => void): void;
  metadata?: Object;
};
```

```ts
function logger<T extends new (...args: any[]) => any>(
  target: T,
  ctx: ClassDecoratorContext
) {
  //дополнительный декоратор
  console.log("init logger"); //запустится при инициализации кода
  return (target: Function) => {
    console.log("run logger");
  };

  // или

  return class extends target {
    constructor(...args: any[]) {
      super(...args);
      console.log("created");
      console.log(this);
    }
  };
}

function addField<T extends new (...args: any[]) => any>(
  target: T,
  ctx: ClassDecoratorContext
) {
  return class extends target {
    // должен совпадать с классом по сигнатуре
    other_id = 35;
  };
}

//декоратор с дополнительными аргументами (1)
function Component(id: number) {
  console.log("init component"); //без явного запуска декоратор запускает функцию
  return (target: Function) => {
    //при доп. аргументах должна возвращать функцию с целевым объектом
    console.log("run component"); //компонент запущен
    target.prototype.id = id; //меняем параметр
  };
}

@logger() //инициализируются сверху вниз но реализация наоборот
@Component(1)
@addField()
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
