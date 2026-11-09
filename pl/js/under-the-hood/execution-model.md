# Agent

То что исполняет js-код, создает хост. Есть у окон, web, shared, service, делится на все связанные iframe и окна сайта. В node у каждого working threads.

Создает агент алгоритм create an agent, делает шаги по созданию записи - создает `[[Signifier]]` - id, `[[CandidateExecution]]` - память (работа с буфером), `[[CanBlock]]` - блокировка другими агентами, привязывает новый event loop, возврат агента

Типы агентов - Similar-origin window agent (страниц, iframe) объект Window , Dedicated worker agent (new Worker()) объект edicatedWorkerGlobalScope, Shared worker agent(new SharedWorker()) объект SharedWorkerGlobalScope, Service worker agent (Service Workers) объект ServiceWorkerGlobalScope, Worklet agent (CSS Paint, звук AudioWorklet) объект WorkletGlobalScope

Созданный агент кладут в кластер - то, где агенты делят память. У Worker каждый раз содается новый агент, у Window распределяется по Site, не по origin. Для вывода в отдельный агент можно использовать `target="_blank"`

Структура:

## set of execution contexts

контекста агента - находится отдельно от execution context stack, у каждого агента свой сет. Главная цель - отладочная

## execution context stack

стек контекстов - содержит контексты в порядке вложенности вызовов. Знает кому вернуть и передать управление, какие EC. Что обычно и называют call stack. Работает по принципу LIFO

## running execution context

исполняемый кадр, вершина стека. в Момент не больше одного (0 или 1). Его можно остановить и передать управлению другому EC. EC не всегда удаляется, а может остаться (например в случае yield или await) Это делается с помощью поля `[[Call]]`

```js
function* gen() {
  console.log("a"); // выполнится на первом .next()
  yield; // тут кадр gen снимается со стека - но не умирает
  console.log("b"); // выполнится на втором .next()
}

const g = gen(); // объект g держит приостановленный кадр gen
g.next(); // a  - дошли до yield, кадр снят со стека
g.next(); // b  - кадр вернулся, доиграл до конца и отпущен: генератор завершён

/* применительно к промисам */

async function f() {
  console.log(1);
  await 0; // здесь кадр f снимается со стека
  console.log(3); // а это уже отдельный Job - стек к тому моменту пуст
}
f(); // вроде ушел со стека, но нет, ссылка на него остается
console.log(2);
// печатает: 1, 2, 3
```

## Agent Record

Настройка агента, у каждого агента один AR. Отвечает за вещи которые не про контекст и не про код. Какой порядок байт относится к агенту. Поля:

- `[[Signifier]]` - id агента
- `[[CanBlock]]` - флаг, может ли блокироваться агент, ждать внешнего события. Выставляется хостом, позволяет executing thread делить поток
- `[[LittleEndian]]` - байты где typed array пишут и читают числа
- `[[IsLockFree1]], [[IsLockFree2]], [[IsLockFree8]]` - lock-free ли атомарные операции в 1,2,8 байт
- `[[CandidateExecution]]` - состояние памяти SharedArrayBuffer
- `[[KeptAlive]]` - объекты удерживаемые WeakRef-ом, которые GC не должен собирать до конца текущей Job
- `[[ModuleAsyncEvaluationCount]]` - счетчик для top-level await

## executing thread

поток, в котором исполняется код (единственное что делится) - поток на котором исполняется код. Может быть использован несколькими агентами. Только нет возможности обмена кадрами

# Execution context

В каждом агенте не больше одного кадра (execution context) в нем лежит все состояние нужно для выполнения кода функции или модуля, который хранит какие переменные. Новый контекст создается при вызове функции, модуля, eval

состоит из компонентов:

- code evaluation state - на каком шаге сейчас
- `[[Function]]` - объект функции, который вызывается, по нему находятся this и super
- `[[Realm]]` - ссылка на Realm Record
- `[[ScriptOrModule]]` - из какого скрипта или модуля код
- `[[LexicalEnvironment]]` - с него начинается поиск имён при выполнении кода. Меняется дважды: при вызове функции (новый функциональный ER) и при каждом входе в блок `{}` (новый блочный ER). Сюда уходят `let`, `const`, `class`
- `[[VariableEnvironment]]` - держит привязки `var`. Меняется только при вызове функции — при входе в блок `{}` не трогается. Именно поэтому `var` "утекает" из блока
- `[[PrivateEnvironment]]` - ссылка на окружение, в нём лежат приватные имена
- `[[Generator]]` - ссылка на объект генератора, есть только у генераторов

Оба поля (`LexicalEnvironment` и `VariableEnvironment`) создаются при вызове функции и изначально указывают на одно окружение. Расходятся при входе в блок: `LexicalEnvironment` переключается на новый блочный ER, `VariableEnvironment` остаётся. Это и объясняет разницу `var` и `let`.

**Механизм блока:**

```
вызов foo():
  LexEnv → funcEnv      VarEnv → funcEnv   ← оба указывают на одно

  вход в блок {}:
    LexEnv → blockEnv → funcEnv            ← переключился на новый ER
    VarEnv → funcEnv                       ← не тронут

  выход из блока:
    LexEnv → funcEnv                       ← вернулся, blockEnv умер
    VarEnv → funcEnv
```

```js
function foo() {
  var a = 1; // → VariableEnvironment (весь foo)
  let b = 2; // → LexicalEnvironment (весь foo)
  {
    var c = 3; // → VariableEnvironment (foo), не блок!
    let d = 4; // → новый LexicalEnvironment для блока
  }
  console.log(a, b, c); // 1 2 3 - c доступна
  // console.log(d); // ReferenceError - d умерла вместе с блочным LexEnv
}
/*
  При входе в {}: LexEnv переключается на новый блочный ER{d:4}
                  VarEnv остаётся прежним ER{a,b,c}
  При выходе из {}: LexEnv возвращается назад
*/
```

## Environment Record

Сущность для замыканий, реализации var, let и области видимости. Связывает идентификаторы с конкретными переменными и функциями, опираясь на структуру кода. По сути это таблица имя-значение. Создается на каждое вычисление, два вызова функции - два разных окружения

Поля:

- `[[OuterEnv]]` - ссылка на внешний ER или null доходит до `[[GlobalEnv]]` при поиске. То есть scope chain - это цепочка вызовов `[[OuterEnv]]`

Окружение - не часть контекста. Живет пока есть хоть одна ссылка на него

```js
function makeCounter() {
  let n = 0;
  return () => ++n;
}
const next = makeCounter(); // кадр makeCounter уже снят со стека и мёртв
next(); // 1
next(); // 2 - n жив: на его окружение до сих пор ссылается возвращённая функция
```

виды Environment Record:

- **`Global Environment Record`** - дно цепочки. Состоит из двух частей:
  - Object Record — для `var` и function declarations; создают свойства на `globalThis`
  - Declarative Record — для `let`, `const`, `class`; **не** создают свойства на `globalThis`
  - Один на весь realm

  ```js
  var x = 1;
  let y = 2;
  globalThis.x; // 1 — var попал в Object Record → globalThis
  globalThis.y; // undefined — let попал в Declarative Record
  ```

- **`Function Environment Record`** - на каждый вызов функции. `[[OuterEnv]]` у нее смотрит не на где вызвали, а где объявили, хранит во внутреннем слоте `[[Environment]]` самого объекта-функции (устанавливается в момент создания функции, не вызова). Именно это делает область видимости лексической а не динамической. Свой this, new.target — здесь, у стрелочных функций нет

- **`Declarative Environment Record`** - базовый вид. Лежит в основе Function и Module, а также создаётся для: блоков `{}`, `catch`, `for`-with-`let`, class body, `eval`

- **`Object Environment Record`** - используется для `with`

- **`Module Environment Record`** - одно на модуль. Импорты — живые привязки (live bindings)

# Realm Record

Отдельная запись, вне Agents но каждый execution context ссылается на него. Состоит из объектов, глобального окружения, весь код, состояние. Состоит из полей:

- `[[Intrinsics]]` - все встроенные объекты (Array, Object, Math, Promise…) так как у каждого iframe свои realm record

  ```js
  const iframe = document.createElement("iframe");
  document.body.append(iframe);
  // массив, созданный конструктором Array из ДРУГОГО realm'а
  const arr = new iframe.contentWindow.Array(1, 2, 3);
  arr instanceof Array; // false - это не «наш» Array, а Array из realm'а iframe
  Array.isArray(arr); // true - а этот способ realm'ы не различает
  ```

- `[[GlobalObject]]`- глобальные объекты, на котором висят свойства Array, setTimeout, var и функции верхнего уровня
- `[[GlobalEnv]]` - глобальное окружение - переменные
- `[[AgentSignifier]]` - id владельца совпадает с `[[Signifier]]` в Agent Record
- `[[TemplateMap]]` - кеш теговых шаблонов
- `[[LoadedModules]]` - кеш модулей
- `[[HostDefined]]` - данные хоста в нем лежит environment settings object

у одного агента может быть несколько RR из-за window.open() или vm.createContext()

settings object состоит из:

- realm execution context - кладет на пустой стек самый первый кадр, вызывает InitializeHostDefinedRealm функцию и кладет на стек новый контекст Function = null ScriptOrModule = null
- module map
- API base URL
- origin и has cross-site ancestor
- policy container - CSP, referrer policy
- cross-origin isolated capability
- time origin

# Jobs

это Abstract Closure без параметров - отложенное вычисление. Запуск, когда стек контекста пуст и не ни одного running контекста. То есть именно он и реализует возврат к нужному контексту. По правилам:

- пустой стек перед запуском
- один за раз
- run-to-completion — каждый Job отрабатывает до нормального завершения (Normal Completion); ошибки обрабатываются внутри
- разбор ошибок

Распределение job ставит хост, различают:

- `HostEnqueuePromiseJob` - для then/catch/finally + await
  - выполняются в порядке fifo
  - к моменту вызова должна быть готова
  - thenable-объекты вызываются внутри Job, не синхронно
- `HostEnqueueTimeoutJob` - для Atomics.waitAsync
- `HostEnqueueGenericJob` - для Atomics.notify
- `HostEnqueueFinalizationRegistryCleanupJob` - вызывает коллбеки FinalizationRegistry для объектов собранных сборщиком мусора

```js
const thenable = {
  then(res) {
    console.log("чужой then");
    res(42);
  },
};
console.log("до");
Promise.resolve(thenable).then((v) => console.log("получили", v));
console.log("после");

// до
// после
// чужой then   ← ушёл в Job, дождался пустого стека
// получили 42
```

# Как это выглядит на практике

Визуализация состояния стека и окружений в момент исполнения.

## Стек и scope chain

```js
const global_x = "global";

function outer() {
  const outer_y = "outer";

  function inner() {
    const inner_z = "inner";
    console.log(global_x, outer_y, inner_z);
    //           ↑ цепочка [[OuterEnv]] поднимается до GlobalEnv
  }

  inner();
}

outer();

/*
Стек (момент выполнения inner):

  ┌─────────────────────────────────────┐  ← running EC
  │  EC: inner                          │
  │  LexEnv → innerEnv                  │
  │  VarEnv → innerEnv                  │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │  EC: outer                          │
  │  LexEnv → outerEnv                  │
  │  VarEnv → outerEnv                  │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │  EC: global script                  │
  │  LexEnv → GlobalDeclEnv             │
  │  VarEnv → GlobalObjEnv              │
  └─────────────────────────────────────┘

Цепочка окружений (scope chain):

  innerEnv{inner_z}
    └─[[OuterEnv]]→ outerEnv{outer_y, inner:fn}
        └─[[OuterEnv]]→ GlobalDeclEnv{global_x}
            └─[[OuterEnv]]→ null
*/
```

## var vs let в блоке

```js
function demo() {
  let a = 1;
  var b = 2;
  {
    let c = 3; // блочный LexEnv
    var d = 4; // уходит наверх в VarEnv функции
    console.log(a, b, c, d); // 1 2 3 4
  }
  console.log(a, b, d); // 1 2 4
  // console.log(c); // ReferenceError
}

/*
До входа в блок:
  LexEnv → funcEnv{a:1} → GlobalEnv
  VarEnv → funcEnv{b:2, d:undefined}  (d уже зарезервирован hoisting-ом)

Внутри блока:
  LexEnv → blockEnv{c:3} → funcEnv{a:1} → GlobalEnv   ← переключился
  VarEnv → funcEnv{b:2, d:4}                           ← не изменился

После блока:
  LexEnv → funcEnv{a:1} → GlobalEnv   ← вернулся назад, blockEnv собран GC
  VarEnv → funcEnv{b:2, d:4}
*/
```

## Замыкание: окружение живёт дольше кадра

```js
function makeAdder(x) {
  // EC makeAdder создан, adderEnv{x} создан
  return function (y) {
    return x + y; // находит x через [[OuterEnv]] → adderEnv
  };
  // EC makeAdder снят со стека, но adderEnv жив — fn держит [[Environment]] → adderEnv
}

const add5 = makeAdder(5); // adderEnv{x:5} жив
const add10 = makeAdder(10); // adderEnv{x:10} — отдельный объект!

add5(3); // 8
add10(3); // 13

/*
  add5.[[Environment]] → adderEnv_1{x:5} → GlobalEnv
  add10.[[Environment]] → adderEnv_2{x:10} → GlobalEnv
  Два вызова makeAdder — два разных окружения
*/
```

# Общая схема

```
┌─────────────────────────────────────────────────────────────┐
│  Agent                                                      │
│                                                             │
│  ┌─────────────────────┐   ┌────────────────────────────┐   │
│  │   Agent Record      │   │  execution context stack   │   │
│  │  [[Signifier]]      │   │                            │   │
│  │  [[CanBlock]]       │   │  ┌──────────────────────┐  │   │
│  │  [[LittleEndian]]   │   │  │ running EC  (вершина)│  │   │
│  │  [[KeptAlive]]      │   │  │  code eval state     │  │   │
│  │  [[CanBlock]]       │   │  │  [[Function]]        │  │   │
│  └─────────────────────┘   │  │  [[Realm]] ──────────┼──┼──→ Realm Record
│                            │  │  [[LexicalEnv]] ─────┼──┼──→ ER_block
│  ┌─────────────────────┐   │  │  [[VariableEnv]] ────┼──┼──→ ER_func
│  │  executing thread   │   │  └──────────────────────┘  │   │
│  │  (может делиться    │   │  ┌──────────────────────┐  │   │
│  │   между агентами)   │   │  │      EC              │  │   │
│  └─────────────────────┘   │  └──────────────────────┘  │   │
│                            │  ┌──────────────────────┐  │   │
│                            │  │   EC (global)        │  │   │
│                            │  └──────────────────────┘  │   │
│                            └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Realm Record
┌─────────────────────────────────────────────────────────────┐
│  [[Intrinsics]]  Array, Object, Promise, Math, ...          │
│  [[GlobalObject]]  window / global / globalThis             │
│  [[GlobalEnv]] ──→ Global Environment Record                │
│                    ├─ Object ER   (var, function decls)     │
│                    └─ Declarative ER (let, const, class)    │
│  [[AgentSignifier]]                                         │
│  [[TemplateMap]], [[LoadedModules]], [[HostDefined]]        │
└─────────────────────────────────────────────────────────────┘

Environment Record chain (scope chain):
  blockEnv{c, d}
    └─[[OuterEnv]]─→ funcEnv{a, b}
        └─[[OuterEnv]]─→ GlobalDeclEnv{x}
            └─[[OuterEnv]]─→ null

Jobs (микрозадачи):
  стек пуст → берём следующий Job из очереди → run-to-completion → повтор
  PromiseJob (then/await) — FIFO, только когда промис resolve/reject
```
