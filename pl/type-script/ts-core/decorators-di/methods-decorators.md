# Декоратор методов

декоратор метода - функция, которая оборачивает логикой метод класса
Не могут быть применены на обычные функции
Не могут быть применены на конструктора функции

Типизация

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
    addInitializer(initializer: () => void): void; //запустится после инициализации
    metadata: Record<PropertyKey, unknown>;
  }
) => Function | void;
```

Пример с переопределением

```ts
function decorator<This, Args extends any[], Return>(
  target: (this: This, ...args: Args) => Return,
  context: ClassMethodDecoratorContext<This>
) {
  return function (this: This, ...args: Args): Return {
    console.log("decorated call:", context.name);
    return target.apply(this, args);
  };
}
```

```ts
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

# bind декоратор

решения проблемы потери контекста при передачи ссылки на метод

```ts
function autobind(
  target: (...args: any[]) => any,
  ctx: ClassMethodDecoratorContext
) {
  ctx.addInitializer(function (this: any) {
    this[ctx.name] = this[ctx.name].bind(this);
  });

  //заменит оригинальную
  return function (this) {
    console.log("execute");
    target.apply(this);
  };
}

class Person {
  name = "Max";
  constructor() {
    //идея -  вариант с bind в constructor
    this.grid = this.grid.bind(this);
  }

  @autobind
  greet() {
    console.log(`hi ${this.name}`);
  }
}

const max = new Person();
const greet = max.greet;

greet();
```

# цепочка декораторов

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
