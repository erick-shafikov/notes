# Прокси (JS Proxy)

Встроенный объект `Proxy` позволяет создать обёртку вокруг другого объекта, перехватывая и переопределяя фундаментальные операции: чтение и запись свойств, вызов функций, проверку наличия ключей и т.д. Логика перехвата задаётся через объект-обработчик (`handler`) с набором ловушек (`traps`).

Отличие от паттерна [Заместитель](./deputy.md): `Proxy` — языковой механизм JS, прозрачный для клиентского кода (не требует явной замены типа), тогда как «Заместитель» — структурный паттерн с явной обёрткой.

Основные ловушки:

- `get(target, prop)` — перехват чтения свойства
- `set(target, prop, value)` — перехват записи свойства
- `has(target, prop)` — перехват оператора `in`
- `apply(target, thisArg, args)` — перехват вызова функции
- `construct(target, args)` — перехват оператора `new`
- `deleteProperty(target, prop)` — перехват `delete`

```ts
// --- Пример 1: валидация при записи ---

const user = { name: "Alice", age: 25 };

const validatedUser = new Proxy(user, {
  set(target, prop, value) {
    if (prop === "age" && typeof value !== "number") {
      throw new TypeError("age must be a number");
    }
    if (prop === "age" && value < 0) {
      throw new RangeError("age must be non-negative");
    }
    target[prop as keyof typeof target] = value;
    return true;
  },
});

validatedUser.age = 30; // OK
// validatedUser.age = -1; // RangeError: age must be non-negative
// validatedUser.age = "old"; // TypeError: age must be a number

// --- Пример 2: логирование доступа к свойствам ---

function withLogging<T extends object>(target: T): T {
  return new Proxy(target, {
    get(obj, prop) {
      console.log(`[get] ${String(prop)}`);
      return Reflect.get(obj, prop);
    },
    set(obj, prop, value) {
      console.log(`[set] ${String(prop)} = ${value}`);
      return Reflect.set(obj, prop, value);
    },
  });
}

const config = withLogging({ host: "localhost", port: 3000 });
config.host; // [get] host
config.port = 8080; // [set] port = 8080

// --- Пример 3: значения по умолчанию (defaultdict) ---

function withDefaults<T>(defaults: T): T {
  return new Proxy({} as T, {
    get(target, prop) {
      return prop in target ? (target as any)[prop] : (defaults as any)[prop];
    },
  });
}

const settings = withDefaults({ theme: "light", lang: "en", debug: false });
console.log(settings.theme); // "light"
console.log((settings as any).unknown); // undefined (из defaults)

// --- Пример 4: перехват вызова функции ---

function memoize<T extends (...args: any[]) => any>(fn: T): T {
  const cache = new Map<string, ReturnType<T>>();

  return new Proxy(fn, {
    apply(target, thisArg, args) {
      const key = JSON.stringify(args);
      if (cache.has(key)) {
        console.log("cache hit");
        return cache.get(key);
      }
      const result = Reflect.apply(target, thisArg, args);
      cache.set(key, result);
      return result;
    },
  }) as T;
}

const expensiveCalc = memoize((n: number) => {
  console.log("computing...");
  return n * n;
});

expensiveCalc(5); // computing... → 25
expensiveCalc(5); // cache hit   → 25
expensiveCalc(6); // computing... → 36
```

Тот же паттерн без встроенного `Proxy` — через классы (GoF Proxy):

```ts
interface Image {
  display(): void;
}

// Реальный объект — дорогая загрузка
class RealImage implements Image {
  private filename: string;

  constructor(filename: string) {
    this.filename = filename;
    this.loadFromDisk();
  }

  private loadFromDisk() {
    console.log(`Loading ${this.filename}`);
  }

  public display() {
    console.log(`Displaying ${this.filename}`);
  }
}

// Прокси — откладывает загрузку до первого вызова display()
class ImageProxy implements Image {
  private filename: string;
  private realImage: RealImage | null = null;

  constructor(filename: string) {
    this.filename = filename;
  }

  public display() {
    if (!this.realImage) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}

const image = new ImageProxy("photo.jpg");
// загрузки ещё не было

image.display(); // Loading photo.jpg → Displaying photo.jpg
image.display(); // Displaying photo.jpg  (загрузка не повторяется)
```
