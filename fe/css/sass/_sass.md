npm i node-sass

```json
// скрипт для запуска
{
  "main": "index.js",
  "scripts": {
    "watch:sass": "node-sass sass/main.scss css/style.css -w",
    "devserver": "live-server",
    "start": "npm-run-all --parallel devserver watch:sass",
    "compile": "node-sass sass/main.scss css/style.comp.css",
    "concat": "concat -o css/style.concat.css css/style.comp.css",
    "prefix": "postcss --use autoprefixer -b 'last versions' css/style.concat.css -o css/style.prefix.css",
    "compress": "node-sass css/style.prefix.css css/style.css --output-style compressed",
    "build": "npm-run-all compile concat prefix compress"
  },
  "devDependencies": {
    "autoprefixer": "^7.1.4",
    "concat": "^1.0.3",
    "node-sass": "^7.0.1",
    "npm-run-all": "^4.1.1",
    "postcss-cli": "^4.1.1"
  }
}
```

sass/main.scss – где находится sass файл
css/style.css – куда компилировать css файл
-w – флаг watch, отслеживает изменения

npm live-server – утилита для запуска проекта

запуск параллельно два окна
