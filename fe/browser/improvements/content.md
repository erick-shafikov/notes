# решения с помощью контента

Вместо пустого экрана загрузки:

- добавить лоудер
- скелетон
- Откладывание отрисовки с помощью content-visibility

```scss
// Может повилять на lcp
.map {
  content-visibility: auto;
  contain-intrinsic-size: 1000px;
}
```

- избегание блокирующих элементов:
- - Встроенные стили и скрипты
- - Вставленные в head элементы link
- - Внешние тэги script без атрибута defer или async.
- Большой размер DOM - отложенная загрузка некоторых блоков
- проблемы на мобильных устройствах Добавить мета-тег viewport (meta name="viewport" content="width=device-width, initial-scale=1")
