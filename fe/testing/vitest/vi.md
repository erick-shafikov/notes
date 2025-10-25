# vi

служит для мокирования

```ts
import { calculator } from "./src/calculator.ts";

vi.mock("./src/calculator.ts", { spy: true });

// позволит работать с импортированной функцией
const result = calculator(1, 2);

expect(result).toBe(3);
expect(calculator).toHaveBeenCalledWith(1, 2);
expect(calculator).toHaveReturned(3);

//или

vi.mock(import("./path/to/module.js"), async (importOriginal) => {
  const mod = await importOriginal(); // type is inferred
  return {
    ...mod,
    // replace some exports
    total: vi.fn(),
  };
});
```

# методы vi.methods

## vo.doMock

```ts
import { beforeEach, test } from "vitest";
import { increment } from "./increment.js";

// вызов оригинальной функции
increment(1) === 2;

let mockedIncrement = 100;

beforeEach(() => {
  // замокали импорт
  vi.doMock("./increment.js", () => ({
    //подмена начнется со 100
    increment: () => ++mockedIncrement,
  }));
});

test("importing the next module imports mocked one", async () => {
  // original import WAS NOT MOCKED, because vi.doMock is evaluated AFTER imports
  expect(increment(1)).toBe(2);
  const { increment: mockedIncrement } = await import("./increment.js");
  // новый динамический импорт предоставляет замокированную функцию
  expect(mockedIncrement(1)).toBe(101);
  expect(mockedIncrement(1)).toBe(102);
  expect(mockedIncrement(1)).toBe(103);
});
```

## vi.mocked

## vi.doUnmock

# мок для функций и объектов

## vi.fn

позволяет создать функцию с нужным поведением

```ts
const getApples = vi.fn(() => 0);

getApples();

expect(getApples).toHaveBeenCalled();
expect(getApples).toHaveReturnedWith(0);

getApples.mockReturnValueOnce(5);

const res = getApples();
expect(res).toBe(5);
expect(getApples).toHaveNthReturnedWith(2, 5);

// или класс
const Cart = vi.fn(
  class {
    get = () => 0;
  }
);

const cart = new Cart();
expect(Cart).toHaveBeenCalled();
```

- vi.isMockFunction

## vi.mockObject

```ts
const original = {
  simple: () => "value",
  nested: {
    method: () => "real",
  },
  prop: "foo",
};

const mocked = vi.mockObject(original);
expect(mocked.simple()).toBe(undefined);
expect(mocked.nested.method()).toBe(undefined);
expect(mocked.prop).toBe("foo");

mocked.simple.mockReturnValue("mocked");
mocked.nested.method.mockReturnValue("mocked nested");

expect(mocked.simple()).toBe("mocked");
expect(mocked.nested.method()).toBe("mocked nested");
```

- vi.clearAllMocks
- vi.resetAllMocks
- vi.restoreAllMocks

## vi.stubEnv

поменяет глобальную переменную

- vi.unstubAllEnvs

## vi.stubGlobal

vi.stubGlobal - для значений globalThis

- vi.unstubAllGlobals

# таймеры

- vi.advanceTimersByTime
- vi.advanceTimersByTimeAsync
- vi.advanceTimersToNextTimer
- vi.advanceTimersToNextTimerAsync
- vi.advanceTimersToNextFrame
- vi.getTimerCount
- vi.clearAllTimers
- vi.getMockedSystemTime
- vi.getRealSystemTime
- vi.runAllTicks
- vi.runAllTimers
- vi.runAllTimersAsync
- vi.runOnlyPendingTimers
- vi.runOnlyPendingTimersAsync
- vi.setSystemTime - дял управления date
- vi.useFakeTimers
- vi.isFakeTimers
- vi.useRealTimers

```ts
let i = 0;
setInterval(() => console.log(++i), 50);

vi.advanceTimersByTime(150);

// log: 1
// log: 2
// log: 3
```

# дополнения

## vi.waitFor

- vi.waitUntil

```ts
import { expect, test, vi } from "vitest";
import { createServer } from "./server.js";

test("Server started successfully", async () => {
  const server = createServer();

  await vi.waitFor(
    //ждет исполнения коллбека
    () => {
      if (!server.isReady) {
        throw new Error("Server not started");
      }

      console.log("Server started");
    },
    {
      // с такими параметрами
      timeout: 500, // default is 1000
      interval: 20, // default is 50
    }
  );
  expect(server.isReady).toBe(true);
});
```

## vi.hoisted

## vi.setConfig
