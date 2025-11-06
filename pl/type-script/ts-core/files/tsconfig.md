установка npm i -g typescript
tsc – команда для работы с tsc
tsc --init – для инициализации, создания tsconfig.json
tsc --watch - hmr для ts

все приложения npm имеют на сайте либо белую иконку, DT. которая говори о том, что нужно установить типы. Если синяя иконка, то типы уже установлены

- npm i -D @types/express – установка типов для express
- tsc – создаст папку dist
- в package.json type:modules лучше поменять на type: common.js
- запуск проекта через node dist/index.js

# tsconfig

- lib - подключение сторонних библиотек типов
- rootDir - обозначить src папку
- outDir - куда будут компилироваться файлы
- noEmitError - генерировать ли js если есть ошибки
