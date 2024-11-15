декораторы - HOC функции, который оборачивают некоторым функционалом методы, аргументы
Декоратор – это функция, оборачивающие свойство логикой. Декораторы делится на декораторы класса, метода, параметра, свойства

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

# Декоратор методов

декоратор метода - функция, которая оборачивает логикой метод класса
Не могут быть применены на обычные функции
Не могут быть применены на конструктора функции

Типизация

```ts
function decorator<This, Args extends any[], Return>(
  target: (this: This, ...args: Args) => Return,
  context: ClassMethodDecoratorContext<
    This,
    (this: This, ...args: Args) => Return,
    descriptor: PropertyDescriptor
  >
): (this: This, ...args: Args) => Return {
  const resultMethod = async function (this: This, ...args: Args): Return {};

  descriptor.value = function newMethod(this: any, ...args:any[]){
    return originalMethod.apply(this, args)
  }

  return descriptor
}
```

```ts
// типизация декоратора метода
type ClassMethodDecorator = (
  target: Function, //ссылка на метод
  context: {
    //описание метода
    kind: "method";
    name: string | symbol; //имя метода
    static: boolean;
    private: boolean;
    access: { get: () => unknown };
    addInitializer(initializer: () => void): void;
    metadata: Record<PropertyKey, unknown>;
  }
) => Function | void;

// создаем декоратор target - ссылка на оригинальную функцию
function myDecorator(target, context) {
  return function (this, ...args) {
    const result = target.apply(this, args);
    return result;
  };
}

class SomeClass {
  @myDecorator
  public someMethod() {}
}
new SomeClass().someMethod();

// для метаданных
SomeClass[Symbol.metadata];
```

Пример 2. Декоратор на отлов ошибки, в методе класса

```ts
function retry(target: any, context: any) {
  const resultMethod = async function (this: any, ...args: any[]) {
    const maxRetryAttempts = 3;
    let lastError = undefined;

    for (let attemptNum = 1; attemptNum <= maxRetryAttempts; attemptNum++) {
      try {
        return await target.apply(this, args);
      } catch (error) {
        lastError = error;

        if (attemptNum < maxRetryAttempts) {
          await sleep(500);
        }
      }
    }

    throw lastError;
  };

  return resultMethod;
}
```

Фабрика декораторов

```ts
function retry2(maxRetryAttempts: number, sleepTime: number) {
  return function (target: any, context: any) {
    const resultMethod = async function (this: any, ...args: any[]) {
      let lastError = undefined;

      for (let attemptNum = 1; attemptNum <= maxRetryAttempts; attemptNum++) {
        try {
          return await target.apply(this, args);
        } catch (error) {
          lastError = error;

          if (attemptNum < maxRetryAttempts) {
            await sleep(sleepTime);
          }
        }
      }

      throw lastError;
    };

    return resultMethod;
  };
}

// применение
@retry2(3, 500)
```

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

# Декоратор параметра

```ts
function Param(
  target: Object,
  propertyKey: string,
  index: number //индекс конкретного параметра
) {
  console.log(propertyKey, index);
}

// @logger
// @Component(1)
// export class User {
//  @Prop id: number;
// @Method

  updatedId(@Param newId: number) {
    this.id = newId;
    return this.id;
  }

// }
// console.log(new User().id);
// console.log(new User().updatedId(2));
```

# Декораторы классов
