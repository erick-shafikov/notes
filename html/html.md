HyperText Markup Language

# стадии отрисовки страницы

- HTML парсится и разбивается на токены
- Токены преобразуются в ноды
- Ноды собираются в dom
- Скачиваются все внешние ресурсы
- Параллельно происходит парсинг CSS, выстраивается CSSOM
- CSSOM и DOM объединяются в Render-tree, в котором скрываются невидимые элементы meta, script, link и добавляются псевдоэлементы before, after
- создается render tree / style calculation
- Переход на стадию Layout, Reflow, Repaint, Composition

![отрисовка страницы](../css/css-assets/page-parsing-stages.png)

Первая стадия это Layout -> Reflow -> Composition

- **layout** отвечает за определение размеров элементов и как их располагать на экране, это самая тяжелая операция для браузера

- **Reflow** – происходит, когда мы изменяем элементы на экране, могут быть изменены как отдельные ветки, так и все дерево именно поэтому документ должен иметь наиболее плоскую структуру. Так же Reflow происходит когда мы снимаем метрики с помощью getBoundingClientRect, offsetLeft, offsetRight. Reflow происходит в основном потоке браузера, там и где крутится event loop, так что на этой стадии JS будет заблокирован

- **Repaint** – на этом этапе браузер обходит layout tree и перекрашивает элементы, как правило Reflow вызывает Repaint

- **Composite** – группировка элементов по слоям, отрисовывает то, что находится не в основном потоке. Все анимации выносятся в отдельный слой, вычисляется в отбельном потоке и JS никак не влияет на него

- **Update** вызывается с помощью getBoundingClientRect()

Вызов repaint

- изменение окна, изменение ориентации
- изменение шрифта
- изменение контента, размера
- добавление/удаление классов стилей
- манипуляции с DOM
- вычисление размеров

# Метрики страницы

- **FCP** – first contentful paint (первая отрисовка) выделять критический css (главное метрика)
- **LCP** – largest contentful paint (отрисовка самого большого элемента) равен FCP без картинок и без критических стилей, выделять height и width для картинки. Убивают видео и фреймы. Можно вставить прямоугольник или картинку, но появляется потенциальная проблема оптимизации
- **TTI** – time to interactive
- **CLS** – Cumulative layout shift (сдвиг от первоначального) правильно выделять критический css
- **TBT** – total blocking time – время, которое нужно для обработки первого действия пользователя
- **SI** – speed index

# Структура кода

```html
<!DOCTYPE html>
<!-- определяет начало файла внутри него <head> и <body> -->
<html>
  <!-- заголовок, его содержимое не показывается не показывается напрямую на странице за исключением -->
  <head>
    <!-- отображайся в левом верхнем углу, является обязательным  -->
    <title>
      <!-- является универсальным, значение description – описание содержимого,
      для атрибута name --> <meta> <!DOCTYPE html> <html> <head> <title>
      Название сайта
    </title>
    <meta name="description" content="Сайт об " />
    <!-- отображается в поисковых системах  -->
    <meta http-equiv="content-type" content="text/html charset = utf-8" />
    <!-- для повышения рейтинга в поисковых системах, можно через пробел или -->
    <meta name="keywords" content="ключевые слова" />
    запятую
  </head>
  <body>
    …
  </body>
</html>
```

- html - корневой элемент
- head - заключает в себе дополнительную информацию
- title - указывает заголовок страницы
- body - для контента страницы

!!! Браузер автоматически сокращает любое количество пробелов до одного
!!! больше 4гб не выделяется на вкладку

<!-- Блочно-строчная модель ------------------------------------------------------------------------------------------------>

# Блочно-строчная модель

## Блочные элементы

Блочные элементы характеризуются тем, что занимают всю ширину, высоту определяет содержимое, всегда начинается с новой строки

blockquote - выделение длинных цитат с отступами по 40рх
hr -горизонтальная линия
pre - моноширинный шрифт, со всеми пробелами обычно любое количество пробелов подряд заменяет на один

!!!могут быть только внутри body

## Строчные

Строчные – являются частью другого элемента, используются для изменения вида или его логического выделения. Можно переопределить в css

!!! Строчные внутри блочных – ок, наоборот нельзя
!!! Блочные начинаются с новой строки
!!! Блочные занимают всю ширину

## Категории данных

- Метаданные: base, link, meta, noscript, script, style и title.
- Основной поток:
- Секционный контент: article, aside, nav, section
- Заголовочный контент: h1-h6, hgroup
- Фразовый контент: все типы ввода
- Встроенный: a, button, details, embed, iframe, label, select, textarea
- Интерактивный: audio[controls=true], img[usemap], input[type!=hidden], menu[type=toolbar],
- ФОРМЫ

## HTML5

- подходящие типы для input, атрибуты валидации

<!-- Блочно-строчная модель -------------------------------------------------------------------------------------------------------------->

# BP. советы по верстке

Стили:

1. Подключить в шапке сайта стили для загрузки первого экрана (критические стили) грузить в шапке, потом перейти к пункту
2. Критические стили можно выделить с помощью утилит
3. Подгруздка стилей по ссылке через JS (создать link и присвоить атрибуты src к стилям) Минус – будет штраф по CLS
4. Выбирать из библиотек нужные куски с помощью препроцессоров
5. Загрузить шрифты можно в двух направлениях – можно загрузить асинхронно, можно выбрать похожий на нужный, что бы предотвратить CLS
6. lazy атрибут для картинок, загрузка шрифтов в двух вариантах (font-display: swap/noswap атрибут, который определяет как загружается шрифт)

Скрипты
Атрибут defer

Виджеты
iframe можно заменить картинкой и загружать по клику
Картинки в webp, выделять размер

# BP. Работа с изображениями

Оптимизация загрузки большого количества изображений

- Зарезервировать место для img с помощью width и height
- Ленивая отрисовка
- использование селекторов srcset (позволяет указать какую картинку загружать в зависимости от VP) и size – работает как медиа-запрос
- ленивая загрузка img loading === lazy
- Размытая заглушка с помощью background-image