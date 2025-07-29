# namespace

Объекты – области кода, которые предоставляю скрытые методы, типы

```ts
namespace NewNameSpace {
  export type TNameSpaceType = {
    // Тип в namespace
  };
  export function funcFromNameSpace() {}
} // Обращение
NewNameSpace.funcFromNameSpace();
// или
import externalFunc = NewNameSpace.funcFromNameSpace; /// <reference path="nameSpaceFile.ts" /> - если в разных модулях содержатся элементы одного namespace
// но тогда в исходном фале нужно будет указать <script src="nameSpaceFile.js" type="text/javascript" /> для всех файлов
```

# ambient namespaces

Для сторонних библиотек создаются файлы

```ts
declare namespace D3 {
  export interface Selectors {
    select: {
      (selector: string): Selection;
      (element: EventTarget): Selection;
    };
  }
  export interface Event {
    x: number;
    y: number;
  }
  export interface Base extends Selectors {
    event: Event;
  }
}
declare var d3: D3.Base;

// namespace - инкапсулирует
export namespace A {
  const a = 5;
  export interface B {
    c: number;
  }
}
```
