# декоратор свойств

```ts
//декоратор для свойства принимает следующие параметры
function Method(
  target: Object,
  propertyKey: string,
  propertyDescriptor: PropertyDescriptor
) {
  console.log(propertyKey);
  const oldValue = propertyDescriptor.value;
  //propertyDescriptor имеет свойства get, set, value и дескрипторы свойств
  propertyDescriptor.value = function (...args: any) {
    return args[0] * 10;
  };
}
```

Декораторы могут быть методом классов

Можно составлять цепочки декораторов

```ts
function decorator1(target: any, context: any) {}
function decorator2(target: any, context: any) {}
function decorator3(target: any, context: any) {}

class SomeClass {
  @decorator1
  @decorator2
  @decorator3
  someMethod() {}
}
// можно объединить

function multipleDecorator(targe: any, context: any) {
  const res1decorator = decorator1(targe, context);
  const res2decorator = decorator2(res1decorator, context);
  const res3decorator = decorator3(res2decorator, context);

  return res3decorator;
}

class SomeClass {
  @multipleDecorator
  someMethod() {}
}
```
