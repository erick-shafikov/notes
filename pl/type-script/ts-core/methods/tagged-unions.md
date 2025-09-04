# Tagged Unions

- принцип работы для случаев { propA?: string } | { propB?: number }
- механизм устраняет необходимость в ручных проверках typeof или in

Прецедент:

```ts
interface Car {
  move(): void;
  startEngine(): void;
}

interface Bicycle {
  move(): void;
  gearCount: number;
}

declare function getVehicle(): Car | Bicycle;

let vehicle = getVehicle();

vehicle.move(); // OK, move() - общее свойство для обоих типов

vehicle.startEngine(); // Ошибка компиляции:

// обход
if ("startEngine" in vehicle) {
  vehicle.startEngine();
}
```

Принцип

```ts
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; side: number }
  | { kind: "rectangle"; width: number; height: number };

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "square":
      return shape.side ** 2;
    case "rectangle":
      return shape.width * shape.height;
    default:
      // Exhaustiveness сhecking
      // Если TypeScript обнаружит необработанный вариант,
      // shape здесь будет иметь тип, отличный от never
      const _exhaustiveCheck: never = shape;
      throw new Error(`Unhandled shape: ${_exhaustiveCheck}`);
  }
}

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; side: number }
  | { kind: "rectangle"; width: number; height: number }
  // выдаст ошибку
  // Type 'triangle' is not assignable to type 'never'
  | { kind: "triangle"; base: number; height: number }; // Новый тип!
```

пример с состоянием

```ts
type LoadingState = { status: "loading" };
type LoadedState<T> = { status: "loaded"; data: T };
type ErrorState = { status: "error"; message: string; errorCode?: number };

type UIState<T> = LoadingState | LoadedState<T> | ErrorState;

function renderUI<T>(state: UIState<T>): React.ReactNode {
  switch (state.status) {
    case "loading":
      return <Spinner />;

    case "loaded":
      // TypeScript гарантирует доступ к state.data
      return <DataView content={state.data} />;

    case "error":
      // Доступны ТОЛЬКО свойства ErrorState
      return <ErrorMessage message={state.message} code={state.errorCode} />;

    default:
      // Защита от будущих изменений
      const _exhaustiveCheck: never = state;
      throw new Error(`Unhandled status: ${_exhaustiveCheck}`);
  }
}
```

пример с api

```ts
type ApiSuccess<T> = {
  response_type: "success";
  data: T;
};

type ApiError = {
  response_type: "error";
  message: string;
  statusCode: number;
};

type ApiResponse<T> = ApiSuccess<T> | ApiError;

async function fetchData<T>(url: string): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return {
        response_type: "error",
        message: `Ошибка HTTP: ${response.statusText}`,
        statusCode: response.status,
      };
    }
    const data: T = await response.json();
    return { response_type: "success", data };
  } catch (error: any) {
    return {
      response_type: "error",
      message: error instanceof Error ? error.message : "Неизвестная ошибка",
      statusCode: 500,
    };
  }
}

// Использование:

type UserData = { id: number; name: string };

async function processUserResponse() {
  const result = await fetchData<UserData>("/api/users/1");

  if (result.response_type === "success") {
    console.log(`Пользователь: ${result.data.name}`); // TypeScript знает, что result.data существует
  } else {
    console.error(
      `Ошибка при получении пользователя: ${result.message} (Код: ${result.statusCode})`
    ); // TypeScript знает, что result.message и statusCode существуют
  }
}
```

тип option

```ts
type None = { kind: "none" };
type Some<T> = { kind: "some"; value: T };
type Option<T> = None | Some<T>;

const getUserName = (user: Option<User>): string => {
  switch (user.kind) {
    case "none":
      return "Гость"; // Используем заглушку когда нет пользователя
    case "some":
      return user.value.name; // TypeScript знает, что user.value существует
  }
};

// Использование:
const loggedInUser: Some<User> = {
  kind: "some",
  value: {
    name: "Иван",
  },
};
const guestUser: None = { kind: "none" };

getUserName(loggedInUser); // "Иван"
getUserName(guestUser); // "Гость"
```

```ts
type Success<T> = { status: "success"; data: T };
type Failure<E> = { status: "failure"; error: E };

type Result<T, E> = Success<T> | Failure<E>;

// Использование:
function parseNumber(input: string): Result<number, string> {
  const num = parseInt(input);
  if (isNaN(num)) {
    return {
      status: "failure",
      error: `Неверный ввод: "${input}" не является числом.`,
    };
  }
  return { status: "success", data: num };
}

const parsed = parseNumber("123");

if (parsed.status === "success") {
  console.log(`Парсинг успешен: ${parsed.data}`);
} else {
  console.error(`Ошибка парсинга: ${parsed.error}`);
}
```
