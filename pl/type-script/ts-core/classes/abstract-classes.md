# абстрактные классы

Абстрактные классы можно только наследовать, но использовать как экземпляр класса – невозможно

```ts
abstract class Base {
  //пример абстрактного класса в отличие от типов - это класс
  print(s: string) {
    console.log(s);
  }
  abstract error(a: string): void; //абстрактный метод абстрактного класса, заставит реализовать этот метод
}
new Base(); //невозможно создать экземпляр этого класса
class BaseExtended extends Base {
  //а наследовать - можно
  error(a: string): void {}
}
new BaseExtended().print("s");
```

## абстракции и implements

Имплементация - взаимодействие интерфейса и класса связать не через два класса, а через адаптер

```ts
interface LoggerService {
  //абстракция как должен работать класс, там должен быть метод log,который ничего не возвращает
  log: (s: string) => void;
}
class Logger implements LoggerService {
  public log(s: string) {
    //без указания типа, по умолчанию тип s будет any, сам метод публичный по умолчанию - public
    console.log(s);
  }
  private error() {} //приватный метод, который не даст доступ l.error(); - Property 'error' is private and only accessible within class 'Logger'.ts(2341)
  private a = ""; //пример приватного
}
const l = new Logger();
l.log("d");
```
