/* ШАБЛОННЫЙ МЕТОД
«Шаблонный метод» определяет каркас выполнения определённого алгоритма, но реализацию самих этапов делегирует
дочерним классам.
«Шаблонный метод» — это поведенческий шаблон, определяющий основу алгоритма и позволяющий наследникам
переопределять некоторые шаги алгоритма, не изменяя его структуру в целом.


Template Method Определяет структуру алгоритма внутри метода, 
при этом поведение зависит от типа класса экземпляра, переданного этому методу. 
Template A = new Student(); Template B = new Teacher(); A.Write(); B.Write();
Visitor Добавляет метод, для работы со структурой объектов. List.Add(Student(“A”)); 
List.Add(Student(“B”)); List.Visit(new VoteVisitor());
 */

abstract class Builder {
  // Шаблонный метод
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

// Output:
// Выполнение Android-тестов
// Линтинг Android-кода
// Создание Android-сборки
// Развёртывание Android-сборки на сервере

const iosBuilder = new IosBuilder();
iosBuilder.build();

// Output:
// Выполнение iOS-тестов
// Линтинг iOS-кода
// Создание iOS-сборки
// Развёртывание iOS-сборки на сервере
