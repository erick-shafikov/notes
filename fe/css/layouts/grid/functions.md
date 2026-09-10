# Функции и единицы grid

## fr (fractional unit)

Относительная единица — доля **свободного пространства** контейнера после вычета фиксированных треков.

```scss
.container {
  display: grid;
  // 200px первая колонка, остаток делится 1:2
  grid-template-columns: 200px 1fr 2fr;
  // три равные колонки
  grid-template-columns: 1fr 1fr 1fr;
  // ряды: первый и последний — свободное пространство, средние — по содержимому
  grid-template-rows: 1fr min-content 6rem 1fr;
}
```

> `fr` вычисляется после фиксированных треков (`px`, `rem`, `min-content`, `max-content`).
> `1fr` = `minmax(auto, 1fr)` — минимум по содержимому.
> `minmax(0, 1fr)` — минимум 0, трек может сжаться до нуля (нужно для предотвращения overflow длинными словами).

## minmax()

Задаёт диапазон размера трека: минимум и максимум. Трек будет между ними.

```scss
.container {
  // трек: минимум 100px, максимум — по содержимому (тянется)
  grid-auto-rows: minmax(100px, auto);
  // трек: минимум по содержимому, максимум — всё свободное место
  grid-template-columns: minmax(min-content, 1fr);
  // трек: не менее 200px, не более 400px
  grid-template-columns: minmax(200px, 400px);
  // sticky footer: header | main (растягивается) | footer
  grid-template-rows: auto minmax(0, 1fr) auto;
}
```

Допустимые значения:
- `min-content` — минимум без overflow (переносит слова)
- `max-content` — размер вмещающий содержимое без переносов
- `auto` — как min: min-content; как max: максимально доступное пространство
- `fr` — только как **максимум**, не как минимум: `minmax(100px, 1fr)` ✓

## repeat()

Избегает дублирования при задании треков:

```scss
.container {
  grid-template-columns: repeat(3, 1fr);           // 1fr 1fr 1fr
  grid-template-columns: repeat(5, 1fr 2fr);        // 10 колонок: 1fr 2fr 1fr 2fr ...
  // именованные линии внутри repeat
  grid-template-columns: repeat(4, [col] 1fr);      // линии col 1..4
  grid-template-columns: repeat(12, [col-start] 1fr);
}
```

### auto-fill vs auto-fit

Оба заполняют ряд максимально возможным числом треков заданного размера. Разница — при количестве элементов меньше чем треков:

| | auto-fill | auto-fit |
|---|---|---|
| Пустые треки | сохраняются (резервируют место) | схлопываются в 0 |
| Элементов мало | не растягиваются | растягиваются на всю ширину |

```scss
.container {
  // фиксированное число колонок вне зависимости от VP
  grid-template-columns: repeat(4, 100px);

  // браузер сам решает сколько колонок поместится
  grid-template-columns: repeat(auto-fill, 100px);

  // авто + минимальный размер + растяжка при широком VP
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));

  // то же, но при малом числе элементов — они растягиваются на всю ширину
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
}
```

## fit-content()

Трек растёт по содержимому, но не превышает переданный аргумент. Формула: `min(max-content, max(auto, argument))`.

```scss
.container {
  // колонка под картинку: растёт по содержимому, но не более 200px
  grid-template-columns: fit-content(200px) 1fr;
  // авто-колонки с ограничением
  grid-auto-columns: fit-content(400px);
  grid-auto-columns: fit-content(5cm);
  grid-auto-columns: fit-content(20%);
}
```

## min() внутри minmax() — anti-overflow паттерн

`minmax(300px, 1fr)` на малых экранах вызывает горизонтальный скролл, если контейнер уже 300px. Решение — `min()`:

```scss
.container {
  // min(300px, 100%): берёт меньшее из двух — трек никогда не выйдет за ширину контейнера
  grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr));
  // то же с 400px — паттерн "auto grid" из скриншота
  grid-template-columns: repeat(auto-fit, minmax(min(400px, 100%), 1fr));
}
```

> Это стандартный современный паттерн для адаптивных grid без медиа-запросов.
