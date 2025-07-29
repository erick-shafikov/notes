# infer

Задаем тип на возврат, который extend тип из которого нужно достать
в опциональной цепочке задаем тип после типа из которого достаем нужное значение

```ts
//пример 1 достать из массива тип элементов
type Flatten<Type> = Type extends Array<infer Item> ? Item : Type; //достаем тип элемента из массива, в противном случае сам тип
const t1: Flatten<string> = "x";
const t2: Flatten<boolean> = false;

type TArrString = [string, boolean];
const array: TArrString = ["x", false];
const t3: Flatten<TArrString> = "y"; //t3: string | boolean
const t4: Flatten<TArrString> = false; //t3: string | boolean

//пример 2 достать из функции тип, который она возвращает
type ReturnType2<Func> = Func extends () => infer X ? X : never; //залезли в тип на возврат
//аргументы
type ArgumentTypes<F extends Function> = F extends (...args: infer A) => any
  ? A
  : never;
type R1 = ReturnType2<() => boolean>; //type R1 = boolean

//пример 3 достать из функции первый аргумент
function runTransaction(transaction: { fromTo: [string, string] }) {
  console.log(transaction);
}

runTransaction(transaction);
//берем функцию, которую принимаем, обозначаем что нужно забрать
type GetFirstArg<T> = T extends (first: infer First, ...args: any[]) => any
  ? First
  : never;

const transaction: GetFirstArg<typeof runTransaction> = {
  fromTo: ["1", "2"],
};

runTransaction(transaction);
```

# BP

## url-parser

```ts
type UrlParamsToUnion<
  URL,
  Acc = never
> = URL extends `${string}:${infer Parameter}/${infer Rest}`
  ? UrlParamsToUnion<Rest, Acc | Parameter>
  : URL extends `${string}:${infer Parameter}`
  ? Acc | Parameter
  : Acc;

// Полученный union тип затем легко преобразовать в тип объекта с помощью следующего кода:
type ParamsUnionToObj<T extends string> = {
  [K in T]: string;
};

type UrlObj<T> = ParamsUnionToObj<UrlParamsToUnion<T>>;
```

```ts
const getUserCommentsURL = "/users/:usersId/comments/:commentsId" as const;

const interpolateURLParameters = <T>(url: T, parameters: UrlObj<T>) => {};

interpolateURLParameters(getUserCommentsURL, {
  usersId: "123",
  commentsId: "1234",
});
```
