# Сортировка слиянием

Разделяем массив пополам до частей длины 0 или 1, затем сливаем отсортированные половины. При слиянии достаточно сравнивать первые ещё не взятые элементы двух частей.

Для `[4, 1, 3, 2]`: половины становятся `[1, 4]` и `[2, 3]`, слияние последовательно выбирает `1, 2, 3, 4`.

```ts
function mergeSort(input: readonly number[]): number[] {
  if (input.length < 2) return [...input];
  const middle = Math.floor(input.length / 2);
  const left = mergeSort(input.slice(0, middle));
  const right = mergeSort(input.slice(middle));
  const result: number[] = [];
  let i = 0;
  let j = 0;
  while (i < left.length && j < right.length) {
    // При равенстве берём слева: сохраняем устойчивость.
    if (left[i] <= right[j]) result.push(left[i++]);
    else result.push(right[j++]);
  }
  while (i < left.length) result.push(left[i++]);
  while (j < right.length) result.push(right[j++]);
  return result;
}

console.log(mergeSort([4, 1, 3, 2, 1])); // [1, 1, 2, 3, 4]
```

Не изменяет вход, устойчива. Время во всех случаях `O(n log n)`, пиковая дополнительная память `O(n)`, глубина стека `O(log n)`. На каждом уровне обрабатывается `O(n)` элементов, уровней `O(log n)`. За всё выполнение выделяется суммарно `O(n log n)` элементов временных массивов, хотя одновременно нужны `O(n)`. Применяется, когда важны устойчивость и гарантированное время.
