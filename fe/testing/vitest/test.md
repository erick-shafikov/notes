# test

```ts
import { test } from "vitest";

test("name", async () => {
  /* ... */
}, 1000); //возможность добавить timeout
```

# контекст

```ts
it("should work", ({ task, expect, skip, annotate, signal, onTestFailed }) => {
  // prints name of the test
  console.log(task.name);
});
```

# методы (test.method)

## extends

Позволяет расширить стандартную функцию

```ts
import { expect, test } from "vitest";

// Дополнительные данные
const todos: number[] = [];
const archive: number[] = [];

interface MyFixtures {
  todos: number[];
  archive: number[];
}

const myTest = test.extend<MyFixtures>({
  todos: async ({ task }, use) => {
    // изменение дополнительных данных
    todos.push(1, 2, 3);
    // запуск тесте с todos
    await use(todos);
    //очистка доп данных
    todos.length = 0;
  },
  //тоже будет передано как archive
  archive,
  // scope-context
  perFile: [({}, use) => use([]), { scope: "file" }],
  perWorker: [({}, use) => use([]), { scope: "worker" }],
});

myTest("add item", ({ todos }: { todos: number[]; archive: number[] }) => {
  expect(todos.length).toBe(3);

  todos.push(4);
  expect(todos.length).toBe(4);
});
```

## skip, skipIf, runIf,only

- skip - Пропуск теста
- skipIf - пропуск по условию
- runIf - запуск по условию

```js
import { assert, test } from "vitest";

test.skip("skipped test", () => {
  // пропуск
});

test("skipped test", (context) => {
  context.skip();
  // пропуск через контекст
});

//skipIf

test.skipIf(isDev)("prod only test", () => {
  // пропуск если isDev === true
});

//runIf

test.runIf(isDev)("dev only test", () => {
  // запуск если isDev === true
});

test.only("test", () => {});
```

# concurrent

Запуск параллельно

```ts
import { describe, test } from "vitest";

// The two tests marked with concurrent will be run in parallel
describe("suite", () => {
  test("serial test", async () => {});
  test.concurrent("concurrent test 1", async () => {});
  test.concurrent("concurrent test 2", async () => {});
});
```

если нужно ссылаться на контекст

```ts
test.concurrent("test 1", async ({ expect }) => {
  expect(foo).toMatchSnapshot();
});
test.concurrent("test 2", async ({ expect }) => {
  expect(foo).toMatchSnapshot();
});
```

# sequential

```ts
import { describe, test } from "vitest";

// with config option { sequence: { concurrent: true } }
test("concurrent test 1", async () => {});
test("concurrent test 2", async () => {});

test.sequential("sequential test 1", async () => {});
test.sequential("sequential test 2", async () => {});

// within concurrent suite
describe.concurrent("suite", () => {
  test("concurrent test 1", async () => {});
  test("concurrent test 2", async () => {});

  test.sequential("sequential test 1", async () => {});
  test.sequential("sequential test 2", async () => {});
});
```

# todo

для тестов, которые еще не написаны

```ts
// An entry will be shown in the report for this test
test.todo("unimplemented test");
```

# each и for

для одного и того же теста но с разными переменными

```ts
import { expect, test } from "vitest";

test.each([
  // список переменных
  // a, b, expected
  [1, 1, 2],
  [1, 2, 3],
  [2, 1, 3],
])("add(%i, %i) -> %i", (a, b, expected) => {
  expect(a + b).toBe(expected);
});

// this will return
// ✓ add(1, 1) -> 2
// ✓ add(1, 2) -> 3
// ✓ add(2, 1) -> 3
```

разница с for как передаются аргументы

```ts
test.for([
  // [a, b, expected]
  [1, 1, 2],
  [1, 2, 3],
  [2, 1, 3],
  // здесь в виде массива
])("add(%i, %i) -> %i", ([a, b, expected]) => {
  expect(a + b).toBe(expected);
});
```

# bench

для измерений времени

```ts
import { bench } from "vitest";

bench(
  "normal sorting",
  () => {
    const x = [1, 5, 4, 2, 3];
    x.sort((a, b) => {
      return a - b;
    });
  },
  { time: 1000 }
);
```

- bench.skip - для пропуска
- bench.todo - для ненаписанных тестов

# describe

позволяет объединить список test, или bench в один файл?можно вкладывать один describe в другой

- describe.skip - пропуск
- describe.skipIf - пропуск по условию
- describe.runIf - запуск по условию
- describe.only - запуск только одной группы тестов
- describe.concurrent - параллельный режим (describe.skip.concurrent, describe.only.concurrent, describe.todo.concurrent)
- describe.sequential - последовательный режим
- describe.shuffle - в произвольном порядке
- describe.each - запуск одного и того же теста с разными переменными

# beforeEach и afterEach

переданный коллбек будет запущен перед (после) каждым тестом

```ts
import { beforeEach } from "vitest";

beforeEach(
  async () => {
    // Clear mocks and add some testing data before each test run
    await stopMocking();
    await addUser({ name: "John" });
  },
  //возможность добавить таймаут
  1000
);
```

# beforeAll, afterAll

запуск перед (после) всех тестов

# onTestFinished

то же самое что и afterEach но для отдельного теста

```ts
import { onTestFinished, test } from "vitest";

test("performs a query", () => {
  const db = connectDb();
  onTestFinished(() => db.close());
  db.query("SELECT * FROM users");
});
```

# onTestFailed

хук для выполнения на ошибку

```ts
import { onTestFailed, test } from "vitest";

test("performs a query", () => {
  const db = connectDb();
  onTestFailed(({ task }) => {
    console.log(task.result.errors);
  });
  db.query("SELECT * FROM users");
});
```
