# Сортировка подсчётом

Считаем число вхождений каждого целого ключа. Накопленные суммы задают границы групп в результате. Это не сортировка сравнениями: эффективность зависит от диапазона ключей `k = max − min + 1`.

```ts
function countingSort(input: readonly number[]): number[] {
  if (input.length === 0) return [];
  let min = input[0];
  let max = input[0];
  for (const value of input) {
    if (!Number.isSafeInteger(value)) throw new RangeError("Нужны безопасные целые числа");
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  const range = max - min + 1;
  // Учебное ограничение памяти: широкий диапазон здесь невыгоден.
  if (!Number.isSafeInteger(range) || range > 1_000_000) {
    throw new RangeError("Слишком широкий диапазон ключей");
  }
  const counts: number[] = new Array(range).fill(0);
  for (const value of input) counts[value - min]++;
  for (let i = 1; i < range; i++) counts[i] += counts[i - 1];

  const result: number[] = new Array(input.length);
  // Обход справа налево сохраняет порядок равных ключей.
  for (let i = input.length - 1; i >= 0; i--) {
    const key = input[i] - min;
    result[--counts[key]] = input[i];
  }
  return result;
}

console.log(countingSort([2, -1, 2, 0, -1])); // [-1, -1, 0, 2, 2]
```

Для примера частоты ключей `[-1, 0, 1, 2]` равны `[2, 1, 0, 2]`, накопленные суммы — `[2, 3, 3, 5]`. Смещение на `min` позволяет работать с отрицательными числами.

Время `O(n + k)`, память `O(n + k)`. Не изменяет вход, устойчива. Для двух значений `0` и `10⁹` огромный массив счётчиков не оправдан. Дробные числа не поддерживаются; округление изменило бы задачу.
