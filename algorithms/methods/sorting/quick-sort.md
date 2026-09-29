# Быстрая сортировка

Выбираем опорное значение (pivot), разделяем элементы относительно него и рекурсивно сортируем части. Важно сохранять все элементы, равные pivot: фильтры только `<` и `>` с добавлением одного pivot теряют дубликаты.

## Вариант с дополнительными массивами

```ts
function quickSort(input: readonly number[]): number[] {
  if (input.length < 2) return [...input];
  const pivot = input[Math.floor(input.length / 2)];
  const less: number[] = [];
  const equal: number[] = [];
  const greater: number[] = [];
  for (const value of input) {
    if (value < pivot) less.push(value);
    else if (value > pivot) greater.push(value);
    else equal.push(value);
  }
  return quickSort(less).concat(equal, quickSort(greater));
}

console.log(quickSort([3, 1, 3, 2, 3])); // [1, 2, 3, 3, 3]
```

Не изменяет вход. Этот вариант устойчив: разбиение сохраняет порядок внутри групп. При сбалансированных разбиениях время `O(n log n)` и пиковая память `O(n)`; в худшем случае время и пиковая память `O(n²)` из-за цепочки несбалансированных вызовов с удерживаемыми массивами. На равных элементах время `O(n)`. Средний элемент по индексу не гарантирует медиану по значению.

## Разбиение Ломуто на месте

```ts
function quickSortInPlace(a: number[]): number[] {
  function partition(left: number, right: number): number {
    const pivot = a[right];
    let boundary = left;
    for (let i = left; i < right; i++) {
      if (a[i] <= pivot) {
        [a[boundary], a[i]] = [a[i], a[boundary]];
        boundary++;
      }
    }
    [a[boundary], a[right]] = [a[right], a[boundary]];
    return boundary; // Pivot уже на окончательной позиции.
  }

  function sort(left: number, right: number): void {
    if (left >= right) return;
    const pivotIndex = partition(left, right);
    sort(left, pivotIndex - 1);
    sort(pivotIndex + 1, right);
  }

  sort(0, a.length - 1);
  return a;
}

console.log(quickSortInPlace([3, 1, 3, 2])); // [1, 2, 3, 3]
```

Этот вариант изменяет массив и неустойчив. Среднее время `O(n log n)` при случайном порядке входа, худшее `O(n²)` — например, на отсортированных или одинаковых элементах при последнем pivot. Память стека в среднем `O(log n)`, в худшем `O(n)`; возможен переполненный стек. Случайный pivot уменьшает вероятность плохих разбиений, а трёхстороннее разбиение полезно при большом числе дубликатов.
