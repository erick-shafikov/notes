```ts
/**
 * Создает тип для строки длины N представляющей шестнадцатеричное число
 * @template N длина строки
 */
type BuildHexString<N extends number> = N extends 0
  ? ""
  : _BuildHexString<N, Digit, [1]>;

/**
 * Промежуточный тип-утилита
 * @template N длина строки
 * @template Result накапливает результат
 * @template Count счетчик цикла, когда Count['length'] === N цикл завершится
 */
type _BuildHexString<
  N extends number,
  Result extends string,
  Count extends unknown[]
> = Count["length"] extends N
  ? Result
  : _BuildHexString<
      N,
      Result | Join<BuildTuple<Digit, [1, ...Count]["length"]>>,
      [1, ...Count]
    >;

/**
 * Преобразует кортеж в шаблонную строку
 * @template T входной кортеж
 */
type Join<T extends string[]> = T extends [
  string,
  ...infer Rest extends string[]
]
  ? `${T[0]}${Join<Rest>}`
  : "";
```

- BuildHexString - Инициализирует счетчик и задает алфавит для служебного типа \_BuildHexString
- BuildHexString: служебный тип, который создает объединение строк и работает со счетчиком Count.
- Join: преобразует кортеж строк в шаблонный строковый литерал.
