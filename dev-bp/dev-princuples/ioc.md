# IoC — Inversion of Control (Инверсия управления)

**IoC** — принцип, при котором не ваш код управляет зависимостями и жизненным циклом объектов, а внешний механизм (фреймворк, контейнер, библиотека).

| | Кто управляет |
|---|---|
| **Без IoC** | ваш код сам создаёт объекты и вызывает нужные функции |
| **С IoC** | кто-то вызывает ваш код и передаёт ему уже готовые зависимости |

---

## IoC ≠ Dependency Injection

Их часто путают, но это разные понятия:

- **IoC** — общий принцип передачи управления внешнему механизму.
- **DI (Dependency Injection)** — один из способов реализовать IoC: зависимости передаются извне, а не создаются внутри класса.

В React вызов компонентов и `useEffect` — это IoC, но не обязательно DI.
В FastAPI `Depends(...)` — одновременно IoC и DI.
Передача `logger` через конструктор в чистом JS — это DI, которая реализует IoC.

---

## 1. Чистый JavaScript

### Без IoC

```js
class Logger {
  log(message) { console.log(message); }
}

class UserService {
  constructor() {
    this.logger = new Logger(); // UserService сам создаёт зависимость
  }
  createUser(name) { this.logger.log(`Created ${name}`); }
}
```

`UserService` контролирует: когда создать `Logger`, какой создать, сколько создать.

### С IoC (через DI)

```js
class UserService {
  constructor(logger) {  // зависимость приходит снаружи
    this.logger = logger;
  }
  createUser(name) { this.logger.log(`Created ${name}`); }
}

const logger = new Logger();
const service = new UserService(logger);
```

Теперь `UserService` ничего не создаёт — только использует то, что ему передали. Можно подменить `Logger` на `FileLogger`, не трогая `UserService`.

---

## 2. Express

Express уже использует IoC на уровне фреймворка: он слушает порт, принимает HTTP, парсит запрос, вызывает middleware и обработчики — ваш код только регистрирует функции.

```js
app.get("/users", (req, res) => { res.send("Hello"); });
// ← вы не вызываете этот обработчик, Express делает это сам
```

### Свой IoC-контейнер для бизнес-логики

```js
class Container {
  constructor() { this.services = new Map(); }
  register(name, instance) { this.services.set(name, instance); }
  get(name) { return this.services.get(name); }
}

// Регистрация зависимостей
const container = new Container();
container.register("userRepository", new UserRepository());
container.register("userService", new UserService(container.get("userRepository")));

// Использование в роуте
app.get("/users", (req, res) => {
  const service = container.get("userService");
  res.json(service.getUsers());
});
```

Получается два уровня IoC: Express управляет HTTP, контейнер управляет бизнес-зависимостями.

---

## 3. React

React использует IoC на уровне фреймворка: вызывает компоненты, `useEffect`, обработчики событий — ваш код пассивен.

```jsx
function Button() { return <button>Hello</button>; }
// React вызывает Button() сам, во время рендера

useEffect(() => { console.log("mounted"); }, []);
// React вызывает callback после рендера, не вы

<button onClick={handleClick}>
// браузер → React → handleClick; вы не вызываете handleClick()
```

### Свой IoC-контейнер через Context

```jsx
class Container {
  constructor() { this.map = new Map(); }
  register(name, value) { this.map.set(name, value); }
  get(name) { return this.map.get(name); }
}

const container = new Container();
container.register("userRepository", new UserRepository());
container.register("userService", new UserService(container.get("userRepository")));

const ContainerContext = createContext(container);

function UsersPage() {
  const container = useContext(ContainerContext);
  const service = container.get("userService");
  const users = service.getUsers();
  // ...
}
```

React управляет рендерингом, контейнер управляет бизнес-логикой. Так же работают InversifyJS и TSyringe.

---

## 4. FastAPI

FastAPI использует IoC через `Depends`: создаёт зависимости, вызывает middleware и endpoints сам.

```python
def get_db():
    return Database()

@app.get("/users")
def users(db = Depends(get_db)):  # FastAPI создаёт db и передаёт его
    ...
```

### Свой IoC-контейнер

```python
class Container:
    def __init__(self):
        self.services = {}

    def register(self, name, service):
        self.services[name] = service

    def get(self, name):
        return self.services[name]

container = Container()
container.register("user_repository", UserRepository())
container.register("user_service", UserService(container.get("user_repository")))

# Интеграция с FastAPI
def get_user_service():
    return container.get("user_service")

@app.get("/users")
def users(service: UserService = Depends(get_user_service)):
    return service.get_users()
```

Цепочка вызовов:

```
FastAPI
  └── Depends(get_user_service)
        └── Ваш контейнер
              └── UserService
                    └── UserRepository
```

---

## Общая схема

```
         Framework IoC
    (Express / React / FastAPI)
               │
               ▼
    вызывает ваш код в нужный момент
               │
               ▼
       Ваш IoC-контейнер
               │
    создаёт сервисы и репозитории
               │
               ▼
         Бизнес-логика
```

Фреймворк управляет жизненным циклом приложения, собственный контейнер — жизненным циклом объектов предметной области. В крупных проектах эти два уровня сосуществуют.
