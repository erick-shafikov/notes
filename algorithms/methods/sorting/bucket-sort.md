# Блочная (карманная) сортировка

Bucket sort распределяет значения по диапазонам — корзинам, сортирует каждую корзину и объединяет их по порядку. Здесь «блочная» означает именно bucket sort; block sort — другое семейство алгоритмов.

Реализация принимает конечные числа из `[0, 1)`. Для `k` корзин индекс равен `floor(value * k)`.

```ts
function bucketSort(input: readonly number[]): number[] {
  const count = Math.max(1, input.length);
  const buckets: number[][] = Array.from({ length: count }, () => []);
  for (const value of input) {
    if (!Number.isFinite(value) || value < 0 || value >= 1) {
      throw new RangeError("Значение должно принадлежать [0, 1)");
    }
    buckets[Math.floor(value * count)].push(value);
  }
  const result: number[] = [];
  for (const bucket of buckets) {
    // Устойчивая сортировка вставками внутри каждой корзины.
    for (let i = 1; i < bucket.length; i++) {
      const value = bucket[i];
      let j = i - 1;
      while (j >= 0 && bucket[j] > value) {
        bucket[j + 1] = bucket[j];
        j--;
      }
      bucket[j + 1] = value;
    }
    for (const value of bucket) result.push(value);
  }
  return result;
}

console.log(bucketSort([0.42, 0.05, 0.9, 0.42])); // [0.05, 0.42, 0.42, 0.9]
```

При четырёх корзинах пример распределяется как `[0.05]`, `[0.42, 0.42]`, `[]`, `[0.9]`.

Не изменяет вход, устойчива. Память `O(n + k)`. Время `O(n + k + Σ mᵢ²)` как верхняя оценка, где `mᵢ` — размер корзины. Ожидаемое время `O(n)` при независимом равномерном распределении и `k = Θ(n)`; худшее `O(n²)`, если много элементов попали в одну корзину в неудобном порядке. Для произвольного диапазона нужна нормализация, с отдельной обработкой максимума и случая `min = max`.
