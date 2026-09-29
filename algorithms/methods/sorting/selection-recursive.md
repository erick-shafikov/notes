# Рекурсивная сортировка выбором

Находим минимум, удаляем его из рабочей копии, рекурсивно сортируем остаток. Это учебный вариант [сортировки выбором](selection-sort.md): рекурсия не улучшает квадратичное время.

```ts
function selectionSortRecursive(input: readonly number[]): number[] {
  const remaining = [...input]; // Исходный массив сохраняется.
  const result: number[] = [];

  function extractMinimum(): void {
    if (remaining.length === 0) return;
    let minIndex = 0;
    for (let i = 1; i < remaining.length; i++) {
      if (remaining[i] < remaining[minIndex]) minIndex = i;
    }
    result.push(remaining.splice(minIndex, 1)[0]);
    extractMinimum();
  }

  extractMinimum();
  return result;
}

const source = [5, 3, 6, 2, 10];
console.log(selectionSortRecursive(source)); // [2, 3, 5, 6, 10]
console.log(source); // [5, 3, 6, 2, 10]
```

Время `O(n²)`, память `O(n)` с учётом копии, результата и стека. Устойчива: выбирается первое вхождение минимума. Пустой массив возвращает `[]`. На больших массивах возможен переполненный стек; итеративный вариант практичнее. Накопление результата вместо цепочки `concat` исключает повторное копирование растущего результата.
