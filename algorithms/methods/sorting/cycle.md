# Циклическая сортировка (Cycle sort)

Сохранённый алгоритм из `cycle.js`. Подсчитываем, сколько элементов в оставшейся части меньше текущего: это определяет его позицию. Записываем туда значение, а вытесненное переносим дальше, пока цикл перестановки не замкнётся.

```ts
function cycleSort(a: number[]): number[] {
  for (let start = 0; start < a.length - 1; start++) {
    let value = a[start];
    let position = start;
    for (let i = start + 1; i < a.length; i++) {
      if (a[i] < value) position++;
    }
    if (position === start) continue;
    // Пропускаем уже занятые равными значениями позиции, не удаляем дубликаты.
    while (a[position] === value) position++;
    [a[position], value] = [value, a[position]];

    while (position !== start) {
      position = start;
      for (let i = start + 1; i < a.length; i++) {
        if (a[i] < value) position++;
      }
      while (a[position] === value) position++;
      [a[position], value] = [value, a[position]];
    }
  }
  return a;
}

console.log(cycleSort([3, 1, 2, 1])); // [1, 1, 2, 3]
```

Для `[3, 1, 2]` переносы образуют цикл: 3 в позицию 2, вытесненная 2 в позицию 1, вытесненная 1 в позицию 0.

Время `O(n²)` даже на отсортированном массиве, память `O(1)`. Изменяет вход, неустойчива. Сильная сторона — небольшое число записей в массив: значение сразу записывается на подходящую итоговую позицию. Может быть полезна, когда запись существенно дороже сравнения.
