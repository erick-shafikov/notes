# опции

- codeSplittingOptions

```js
// vite.config.ts
import { defineConfig } from "vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

export default defineConfig({
  plugins: [
    tanstackRouter({
      autoCodeSplitting: true,
      codeSplittingOptions: {
        splitBehavior: ({ routeId }) => {
          // For all routes under /posts, bundle the loader and component together
          if (routeId.startsWith("/posts")) {
            return [["loader", "component"]];
          }
          // All other routes will use the `defaultBehavior`
        },
        defaultBehavior: [
          ["loader"], // The loader will be in its own chunk
          ["component"],
          // ... other component groupings
        ],
      },
    }),
  ],
});
```

- routeDirectory (обязательно) - ./src/routes по умолчанию
- generatedRouteTree (обязательно) - ./src/routeTree.gen.ts
- virtualRouteConfig - undefined для виртуальных маршрутов
- routeFilePrefix - '' по умолчанию, префикс для роутов
- routeFileIgnorePrefix- По умолчанию "-" префикс для игнорируемых
- routeFileIgnorePattern - паттерн для название файлй, который будет проигнорирован
- indexToken - index по умолчанию, название файла, который будет главным для пути в папке
- routeToken - route по умолчанию, то есть убирает route часть из пути
- quoteStyle - single
- semicolons - false, если true генерация с ";"
- APIBase
- autoCodeSplittingOptionsSplitting - false по умолчанию, разделение кода
- disableTypes - false, отключить генерацию типов
- addExtensions - false, добавляет расширение в название маршрутов
- disableLogging - false, отключить ведение журнала
- routeTreeFileHeader - добавить содержимое в начале пути

```js
//по умолчанию
[
  "/* eslint-disable */",
  "// @ts-nocheck",
  "// noinspection JSUnusedGlobalSymbols",
];
```

- routeTreeFileFooter - [] добавит в конец файла сгенерированного
- enableRouteTreeFormatting - true, форматирование в сгенерированных файлах
- tmpDir - директория для временно сгенерированных фалов
