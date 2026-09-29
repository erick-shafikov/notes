# Поразрядная сортировка

LSD-вариант сортирует от младшего разряда к старшему. Каждый проход обязан быть устойчивым, чтобы порядок, полученный по младшим разрядам, сохранился внутри одинакового старшего разряда.

Ниже основание 256: безопасные неотрицательные целые числа сортируются по байтам. Деление используется вместо побитовых операторов, которые в JavaScript сужают число до 32 бит.

```ts
function radixSort(input: readonly number[]): number[] {
  let result = [...input];
  let max = 0;
  for (const value of result) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError("Нужны неотрицательные безопасные целые числа");
    }
    max = Math.max(max, value);
  }
  const base = 256;
  for (let place = 1; place <= max; place *= base) {
    const counts: number[] = new Array(base).fill(0);
    const digit = (value: number): number => Math.floor(value / place) % base;
    for (const value of result) counts[digit(value)]++;
    for (let i = 1; i < base; i++) counts[i] += counts[i - 1];
    const next: number[] = new Array(result.length);
    for (let i = result.length - 1; i >= 0; i--) {
      next[--counts[digit(result[i])]] = result[i];
    }
    result = next;
  }
  return result;
}

console.log(radixSort([256, 2, 257, 0, 2])); // [0, 2, 2, 256, 257]
```

После первого прохода: `[256, 0, 257, 2, 2]`, после второго — итоговый порядок. Нули и пустой массив поддерживаются; отрицательные и дробные значения отклоняются.

Время `O(d(n + b))`, память `O(n + b)`, где `d` — число разрядов, `b` — основание. С учётом первоначального копирования/проверки даже при `max = 0` время `O(n)`. При фиксированной разрядности время линейно по `n`. Не изменяет вход, устойчива. Для знаковых чисел нужна отдельная обработка знака; простой разворот отрицательной части нарушает устойчивость равных ключей.
