# Сортировка выбором

На каждом шаге находим минимум неотсортированного суффикса и ставим его в начало этого суффикса. Это сортировка выбором, а не пузырьковая: здесь выбирается минимум, а не многократно сравниваются соседи.

```ts
function selectionSort(a: number[]): number[] {
  for (let i = 0; i < a.length - 1; i++) {
    let minIndex = i;
    for (let j = i + 1; j < a.length; j++) {
      if (a[j] < a[minIndex]) minIndex = j;
    }
    if (minIndex !== i) [a[i], a[minIndex]] = [a[minIndex], a[i]];
  }
  return a;
}

// Вариант прежнего примера: извлекаем минимум из копии массива.
function selectionSortCopy(input: readonly number[]): number[] {
  const remaining = [...input];
  const result: number[] = [];
  while (remaining.length > 0) {
    let minIndex = 0;
    for (let i = 1; i < remaining.length; i++) {
      if (remaining[i] < remaining[minIndex]) minIndex = i;
    }
    result.push(remaining.splice(minIndex, 1)[0]);
  }
  return result;
}

console.log(selectionSort([5, 3, 6, 2, 10])); // [2, 3, 5, 6, 10]
console.log(selectionSortCopy([2, 1, 2])); // [1, 2, 2]
```

Основной вариант изменяет массив, требует `O(1)` дополнительной памяти и делает не более `n − 1` обменов. Во всех случаях время `O(n²)`. Неустойчив: обмен в `[2a, 2b, 1]` даёт `[1, 2b, 2a]`.

Вариант с копией не изменяет вход, устойчив благодаря выбору первого минимума, но требует `O(n)` памяти; `splice` дополнительно сдвигает элементы. Время остаётся `O(n²)`. См. также [рекурсивный вариант](selection-recursive.md).
