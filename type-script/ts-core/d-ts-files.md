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
