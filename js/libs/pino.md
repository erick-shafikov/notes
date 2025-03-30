# настройки конструктора

```js
import pino from "pino";

const logger = pino(
  //setting
  {
    name: undefined, //string - имя логгера
    level: "info", //минимальный уровень отображения
    //сравнение уровней логов
    levelComparison: "ASC", //"DESC"
    levelComparison: function (current, expected) {
      //если функция
      return current >= expected;
    },
    customLevels: undefined,
    customLevels: {
      foo: 35,
    },
    useOnlyCustomLevels: false,
    depthLimit: 5, //количество вложенных
    //mixin
    mixin: undefined,
    mixin: (mergeObject, level, loggerInstanceOrLoggerChild) => {
      return JSONObject;
    },
    mixinMergeStrategy: (mergeObject, mixinObject) => {
      // стратегия по умолчанию
      return Object.assign(mixinMeta, mergeObject);
      //return Object.assign({}, mergeObject, mixinObject)
    },
    //исключения
    redact: undefined,
    redact: [
      "key",
      "path.to.key",
      "stuff.thats[*].secret",
      'path["with-hyphen"]',
    ], //пути до ключей
    redact: {
      paths: ["path.to.redact.keys"], //пути до ключей
      censor: "[Redacted]", //строка замены
      censor: (value, path) => {
        // path === "path.to.redact.keys"
      },
      remove: false, //удалять редактированные ключи
    },
    //хуки
    hooks: {
      logMethod: (args, method, level) => {
        //args - переданные в логгер аргументы
        //method - логгер для вызова
        if (inputArgs.length >= 2) {
          const arg1 = inputArgs.shift();
          const arg2 = inputArgs.shift();
          return method.apply(this, [arg2, arg1, ...inputArgs]);
        }

        return method.apply(this, inputArgs);
      },
      streamWrite: (s) => {
        return s.replaceAll("sensitive-api-key", "XXX");
      },
    },
    formatters: {
      //для форматирования поля уровня
      level(label, number) {
        return { level: number };
      },
      //для форматирования поля связывания
      bindings(bindings) {
        //{ pid, hostname }
        return { pid: bindings.pid, hostname: bindings.hostname };
      },
      // объект логирования
      log(object) {
        return object;
      },
    },
    serializers: pino.stdSerializers.err, //по умолчанию
    msgPrefix: undefined, //ключ для второго аргумента
    base: { pid: process.pid, hostname: os.hostname() }, //базовые поля логгера
    enabled: true,
    crlf: false,
    timestamp: true,
    timestamp: () => `,"time":"${new Date(Date.now()).toISOString()}"`,
    messageKey: "msg", //ключ для второго строкового параметра
    errorKey: "err",
    nestedKey: null,
    nestedKey: "payload", //для группировки данных
    browser: {},
    transport: { target: "/absolute/path/to/my-transport.mjs" },
    onChild: (instance) => {}, // сработает при создании каждого нового child
    destination: {
      dest: "/log/path",
      sync: false,
      [Symbol.for("pino.metadata")]: {},
    },
  },

  //destination
  {}
);
```

## mixins

При передачи аргументов в первой логгер поле description будет и во втором, если это логи одного уровня

```js
const mixin = {
  appName: "My app",
};

const logger = pino({
  mixin() {
    return mixin;
  },
});

logger.info(
  {
    description: "Ok",
  },
  "Message 1"
);
// {"level":30,...,
// "appName":"My app","description":"Ok","msg":"Message 1"}
logger.info("Message 2");
// {"level":30,...
// "appName":"My app","description":"Ok","msg":"Message 2"}
// Во втором логе тоже есть description: "Ok"
```

<!-- свойства экземпляра ----------------------------------------------------------------------------------------------------------->

# свойства экземпляра

## logger.trace(), info...

```js
logger
  //debug, info, warn, error, fatal
  .trace(
    // mergingOBject,
    {},
    // message,
    "some string for msg setting keys"
    // [interpolationValues]
  );
```

# logger.silent();

```js
logger.silent();
```

# logger.child()

позволяет создать логгер с состоянием

```js
const childLogger = logger.child(
  //bindings
  {
    // object который будет перенаправлен всем childLogger
  },
  // [options]
  {
    level,
    msgPrefix: undefined,
    msgPrefix: "prefix",
    redact: {},
    redact: [],
    serializers: {},
  }
);
```

```js
const child = logger.child({ MIX: { IN: "always" } });
child.info("hello");
// {"level":...,"msg":"hello","pid":64849,"hostname":"x","MIX":{"IN":"always"}}
child.info("child!");
// {"level":...,"msg":"child!","pid":64849,"hostname":"x","MIX":{"IN":"always"}}
```

## logger.bindings()

```js
const child = logger.child({ foo: "bar" });
console.log(child.bindings());
// { foo: 'bar' }
const anotherChild = child.child({ MIX: { IN: "always" } });
console.log(anotherChild.bindings());
// { foo: 'bar', MIX: { IN: 'always' } }
```

## logger.setBindings(bindings)

не перезапишет, может привести к дублированию

```js
logger.setBindings(
  // bindings
  {}
);
```

## logger.flush(cb)

очищает логгер

## logger.isLevelEnabled(level)

## logger.levelVal

## logger.levels

## событийная модель

- levelLabel – the new level string, e.g trace
- levelValue – the new level number, e.g 10
- previousLevelLabel – the prior level string, e.g info
- previousLevelValue – the prior level number, e.g 30
- logger – the logger instance from which the event originated

```js
const logger = require("pino")();
logger.on("level-change", (lvl, val, prevLvl, prevVal) => {
  console.log("%s (%d) was changed to %s (%d)", prevLvl, prevVal, lvl, val);
});
logger.level = "trace"; // trigger event
```

## logger.version

<!-- статические поля  ----------------------------------------------------------------------------------------------------------------------->

# статические поля

## pino.destination()

```js
const pino = require("pino");
const logger = pino(pino.destination("./my-file"));
const logger2 = pino(pino.destination());
const logger3 = pino(
  pino.destination({
    dest: "./my-file",
    minLength: 4096, // Buffer before writing
    sync: false, // Asynchronous logging, the default
  })
);
const logger4 = pino(
  pino.destination({
    dest: "./my-file2",
    sync: true, // Synchronous logging
  })
);
```

## pino.transport(options)

```js
const pino = require("pino");
const transport = pino.transport({
  targets: [
    {
      level: "info",
      target: "pino-pretty", // must be installed separately
    },
    {
      level: "trace",
      target: "pino/file",
      options: { destination: "/path/to/store/logs" },
    },
  ],
});
pino(transport);
```
