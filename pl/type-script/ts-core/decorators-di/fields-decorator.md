# декоратор свойств

```ts
//декоратор для свойства принимает следующие параметры
function fieldDecorator(value: any, context: ClassFieldDecoratorContext) {
  console.log("field =", context.name);

  //новое значение
  return (initialValue: any) => initialValue;
}

type ClassFieldDecoratorContext = {
  kind: "field";
  name: string; //имя поля
  static: boolean;
  private: boolean;
  acc: {
    has: Function;
    get: Function;
    set: Function;
  };
  metadata?: any;
  addInitializer: Function;
};
```

```ts
// value === undefined так как при инициализации
function fieldDecorator(value: undefined, context: ClassFieldDecoratorContext) {
  console.log("field =", context.name);
}

class Person {
  @fieldDecorator
  name = "Max";
}
```
