d.ts файлы предназначены для типизации сторонних библиотек или для объявления глобальной типизации
Можно активировать автоматическое создание d.ts файлов, с помощью опции declaration: true в ts.config файле

# ambient declaration

```ts
//lib-global.d.ts файл

//Объявляет
declare namespace LibName {
  function someFunc(args: ArgsTypes): ReturnType;
}

// для библиотек
export default LibName;
```

# triple-slash

можно в файле добавить ссылку на типизацию

```ts
/// <reference path="./lib-path"
```

Файловая структура project/:

- src/
- - index.ts
- types/
- - globals.d.ts
- - extra.d.ts

Глобальный типы

```ts
// файл: types/globals.d.ts
declare global {
  const API_URL: string;
}
```

экстра типы + глобальные

```ts
// файл: types/extra.d.ts
/// <reference path="./globals.d.ts" />

declare global {
  function logApiUrl(): void;
}
```

подключение типов

```ts
// файл: src/index.ts
/// <reference path="../types/extra.d.ts" />

logApiUrl(); // работает
console.log(API_URL); // работает
```

# объявление в window

```ts
export {};

interface DeepAnalytics {
  reachGoal: (goal: string) => void;
}

declare global {
  interface Window {
    deepAnalytics: DeepAnalytics;
  }
}
```

# сторонние библиотеки

Устанавливаются в node_modules/@types
