# Классы и generic

```ts
// пример 1
class MyClass {
  static a = "1"; //ненужно инициализировать
  static {} //статический блок инициализации
}

MyClass.a = "asd";

class AnotherClass<T> {
  a: T;
}
const b = new AnotherClass<string>(); //b.a будет строкой

// пример 2
class Resp<D, E> {
  //класс из двух полей
  data?: D;
  error?: E;
  constructor(data?: D, error?: E) {
    if (data) {
      this.data = data;
      this.error = error;
    }
  }
}
const res = new Resp<string, number>("data", 123);
class HTTPResp<F> extends Resp<string, number> {
  //расширяем с жоп generic
  code?: F;
  setCode(code: F) {
    this.code = code;
  }
}
const res2 = new HTTPResp();
```
