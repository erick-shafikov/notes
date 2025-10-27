# expect

Использование методов:

```ts
import { expect } from "vitest";

expect(/*выражения для тестирования*/).method();
```

для создания утверждения, замена test, методы:

- assert - сравнение

```ts
const animal: Animal = { __type: "Dog", bark: () => {} };

expect.assert(animal.__type === "Dog");
expect(animal.bark()).toBeUndefined();
```

- soft - тоже самое что и expect, не остановит тесты пока они не выполнятся даже в случае ошибки, все ошибки будут в консоли
- poll - будет пробовать пока не истечет таймаут

- Логические:
- - not

```ts
import { expect, test } from "vitest";

const input = Math.sqrt(16);

expect(input).not.to.equal(2); // chai API
expect(input).not.toBe(2); // jest API
```

- - toBe - сравнение
- - toBeCloseTo - сравнение с округлением
- - toBeDefined
- - toBeUndefined
- - toBeTruthy
- - toBeFalsy
- - toBeNull
- - toBeNullable
- - toBeNaN
- - toBeOneOf

```ts
import { expect, test } from "vitest";

test("fruit is one of the allowed values", () => {
  expect(fruit).toBeOneOf(["apple", "banana", "orange"]);
});
```

- проверка наследования:
- - toBeTypeOf
- - toBeInstanceOf

- числовые выражения:
- - toBeGreaterThan

```ts
test("have more then 10 apples", () => {
  expect(getApples()).toBeGreaterThan(10);
});
```

- - toBeGreaterThanOrEqual
- - toBeLessThan
- - toBeLessThanOrEqual
- - toEqual

- Объекты
- - toStrictEqual
- - toContain
- - toContainEqual
- - toHaveLength
- - toHaveProperty
- - toMatch - regex
- - toMatchObject
- toThrowError
- toMatchFileSnapshot
- toThrowErrorMatchingSnapshot
- toThrowErrorMatchingInlineSnapshot

- для работы с функциями:
- - toHaveBeenCalled
- - toHaveBeenCalledTimes
- - toHaveBeenCalledWith
- - toHaveBeenCalledBefore
- - toHaveBeenCalledAfter
- - toHaveBeenCalledExactlyOnceWith
- - toHaveBeenLastCalledWith
- - toHaveBeenNthCalledWith
- - toHaveReturned
- - toSatisfy

```ts
import { describe, expect, it } from "vitest";

const isOdd = (value: number) => value % 2 !== 0;

describe("toSatisfy()", () => {
  it("pass with 0", () => {
    expect(1).toSatisfy(isOdd);
  });

  it("pass with negation", () => {
    expect(2).not.toSatisfy(isOdd);
  });
});
```

- Промисы:
- - toHaveLastResolvedWith
- - resolves - для того что бы убрать бойлерплейт кода с промисами
- - rejects

- Утилиты:
- - expect.assertions - дял работы с контролем количества тестов

```ts
import { expect, test } from "vitest";

async function doAsync(...cbs) {
  await Promise.all(cbs.map((cb, index) => cb({ index })));
}

test("all assertions are called", async () => {
  // должно быть 2 теста
  expect.assertions(2);
  function callback1(data) {
    expect(data).toBeTruthy();
  }
  function callback2(data) {
    expect(data).toBeTruthy();
  }

  await doAsync(callback1, callback2);
});
```

- - для учета тестов:
- - - expect.hasAssertions - был ли тест
- - - expect.unreachable - части кода, которые не будут достигнуты

- - для результатов теста:

- - - expect.anything

```ts
import { expect, test } from "vitest";

test('object has "apples" key', () => {
  expect({ apples: 22 }).toEqual({ apples: expect.anything() });
});
```

- - - expect.any
- - - expect.closeTo
- - - expect.arrayContaining
- - - expect.objectContaining
- - - expect.stringContaining
- - - expect.stringMatching
- - - expect.schemaMatching

- - для написания кастомных тестов:
- - - expect.addSnapshotSerializer
- - - expect.extend
- - - expect.addEqualityTesters
