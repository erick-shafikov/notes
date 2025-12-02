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
    addInitializer(initializer: () => void): void;
    metadata: Record<PropertyKey, unknown>;
  }
) => Function | void;
```

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
