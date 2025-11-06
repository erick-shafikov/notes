# нюансы поведения

```ts
const stringArray = ["x", "y"];
stringArray.push(1); //error
```

# типизация массивов

```ts
// порядок имеет значения, если разные типы
const numTriplet: [number, number, number] = [7, 7, 7];
numTriplet.length; //3
numTriplet.pop();
numTriplet.pop();
numTriplet.pop();
numTriplet.length; //3 ts не видит, что мы вытащили все элементы из, лечится с помощью readonly
```

```ts
type Array2<T> = [T, ...T[]]; //массив с одним обязательным элементом
type Array2<T> = [T, T, ...T[]]; //массив с двумя обязательными элементом
```

# Создает кортеж заданной длины

```ts
/**
 * Создает кортеж заданной длины
 * @template T типа элементов кортежа
 * @template N длина кортежа
 */
type BuildTuple<T, N extends number> = _buildTuple<T, N>;
type _buildTuple<
  T,
  N extends number,
  Result extends T[] = []
> = Result["length"] extends N ? Result : _buildTuple<T, N, [T, ...Result]>;
```
