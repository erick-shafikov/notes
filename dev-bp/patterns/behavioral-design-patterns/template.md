# ШАБЛОННЫЙ МЕТОД

«Шаблонный метод» определяет каркас выполнения определённого алгоритма, но реализацию самих этапов делегирует дочерним классам. Это поведенческий шаблон, определяющий основу алгоритма и позволяющий наследникам переопределять некоторые шаги алгоритма, не изменяя его структуру в целом.

Template Method (Шаблонный метод) - Определяет структуру алгоритма внутри метода, при этом поведение зависит от типа класса экземпляра. `Template A = new Student(); Template B = new Teacher(); A.Write(); B.Write();`

```ts
abstract class Builder {
  // Шаблонный метод — определяет порядок шагов
  public build() {
    this.test();
    this.lint();
    this.assemble();
    this.deploy();
  }

  abstract test(): void;
  abstract lint(): void;
  abstract assemble(): void;
  abstract deploy(): void;
}

class AndroidBuilder extends Builder {
  public test() {
    console.log("Running android tests");
  }

  public lint() {
    console.log("Linting the android code");
  }

  public assemble() {
    console.log("Assembling the android build");
  }

  public deploy() {
    console.log("Deploying android build to server");
  }
}

class IosBuilder extends Builder {
  public test() {
    console.log("Running ios tests");
  }

  public lint() {
    console.log("Linting the ios code");
  }

  public assemble() {
    console.log("Assembling the ios build");
  }

  public deploy() {
    console.log("Deploying ios build to server");
  }
}

const androidBuilder = new AndroidBuilder();
androidBuilder.build();
// Running android tests
// Linting the android code
// Assembling the android build
// Deploying android build to server

const iosBuilder = new IosBuilder();
iosBuilder.build();
// Running ios tests
// Linting the ios code
// Assembling the ios build
// Deploying ios build to server
```
