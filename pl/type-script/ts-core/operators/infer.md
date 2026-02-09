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
```

# infer и функции

```ts
//пример 2 достать из функции тип, который она возвращает
type ReturnType2<Func> = Func extends (...args: any[]) => infer X ? X : never; //залезли в тип на возврат

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

## tuples

### достать все кроме первого элемента из массива

задачи из типа Variadic Tuple Hell

```ts
type Tail<T extends any[]> = T extends [any, ...infer R] ? R : never;
type A = Tail<[1, 2, 3]>;
// [2, 3]
```

### перевернуть массив

```ts
type Reverse<T extends any[]> = T extends [infer H, ...infer R]
  ? [...Reverse<R>, H]
  : [];
type R = Reverse<[1, 2, 3]>;
// [3, 2, 1]
```

### zip двух tuple

```ts
type Zip<A extends any[], B extends any[]> = A extends [infer AH, ...infer AR]
  ? B extends [infer BH, ...infer BR]
    ? [[AH, BH], ...Zip<AR, BR>]
    : []
  : [];
type Z = Zip<[1, 2], ["a", "b"]>;
// [[1, 'a'], [2, 'b']]
```

### tuple + overload API

```ts
type Pipe<T extends any[]> = T extends [infer A, infer B, ...infer R]
  ? Pipe<[(x: A) => B, ...R]>
  : T;
```

## строки

### SnakeToCamel

```ts
type SnakeToCamel<S extends string> = S extends `${infer H}_${infer R}`
  ? `${H}${Capitalize<SnakeToCamel<R>>}`
  : S;

type C = SnakeToCamel<"user_profile_id">;
// 'userProfileId'
```

### url-parser 1

```ts
type ParseRoute<T extends string> = T extends `${infer Param}/${infer Rest}`
  ? Param | ParseRoute<Rest>
  : T;

type R = ParseRoute<"users/:id/posts/:postId">;
// 'users' | ':id' | 'posts' | ':postId'
```

### url-parser 2

```ts
type UrlParamsToUnion<
  URL,
  Acc = never,
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
