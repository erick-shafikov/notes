# namespace

Объекты – области кода, которые предоставляю скрытые методы, типы. При компиляции собираются в один объект

```ts
// nameSpaceFile.ts
namespace NewNameSpace {
  // если не добавить export, то TNameSpaceType будет доступен только внутри NewNameSpace
  export type TNameSpaceType = {
    // Тип в namespace
  };
  export function funcFromNameSpace() {}
}
```

```ts
/// <reference path="nameSpaceFile.ts" />
// если в разных модулях содержатся элементы одного namespace
// Обращение
funcFromNameSpace();
// или
import externalFunc = funcFromNameSpace;
// но тогда в исходном файле нужно будет указать <script src="nameSpaceFile.js" type="text/javascript" /> для всех файлов
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
