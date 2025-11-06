# строка минимальной длины

```ts
type AtLeastTwoCharsType = `${any}${string}${string}`;
type AtLeastThreeCharsType = `${any}${string}${string}${string}`;
```

# snake -> camel

```ts
type FromSnakeToCamelCase<T extends string> =
  T extends `${infer Word}_${infer Rest}`
    ? `${Capitalize<Word>}${FromSnakeToCamelCase<Rest>}`
    : T;
```

через хвостовую рекурсию

```ts
type FromSnakeToCamelCase<
  T extends string,
  S extends string = ""
> = T extends `${infer Word}_${infer Rest}`
  ? FromSnakeToCamelCase<Capitalize<Rest>, `${S}${Word}`>
  : `${S}${T}`;
```
