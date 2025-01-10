# Комбинаторы

Комбинаторы

- [запятая – группировка](#комбинатор-запятая-)
- [пробел – выбор всех потомков](#селектор-потомка-пробел)
- [> - дочерние (непосредственно прямые потомки)](#дочерний-комбинатор--знак-больше)
- [~ - выбор всех одноуровневых](#комбинатор-всех-соседних--тильда)
- [+ - выбор первого соседнего элемента](#соседние-селекторы--знак-плюс)

## Комбинатор запятая ,

Позволяет сгруппировать определение стилей

```html
<style>
  h1 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 160%;
    color: #003;
  }
  h2 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 135%;
    color: #333;
  }
  h3 {
    /* дублирование кода */
    font-family: Arial, Helvetica, sans-serif;
    font-size: 120%;
    color: #900;
  }
  P {
    /* дублирование кода */
    font-family: Times, serif;
  }
</style>
```

Избежать дублирование кода можно

```html
<style>
  h1,
  h2,
  h3 {
    font-family: Arial, Helvetica, sans-serif;
  }
  h1 {
    font-size: 160%;
    color: #003;
  }
  h2 {
    font-size: 135%;
    color: #333;
  }
  h3 {
    font-size: 120%;
    color: #900;
  }
</style>
```

## Селектор потомка (пробел)

Будет применятся для всех вложенных элементов, для всех дочерних

При использовании идентификаторов и классов – позволяет установить стиль внутри определенного класса

```scss
span {
  background-color: white;
}
div span {
  background-color: DodgerBlue;
}
```

```html
<p></p>
<div>
  <span
    >Span 1 (применится)
    <span>Span 2 (применится)</span>
  </span>
</div>
<span>Span 3 (нет)</span>
```

сочетания

```scss
.level11 {
  front-size: 1em;
}
.level12 {
  front-size: 1.2em;
}

a.tag {
  color: #62348;
}
```

```html
<a href="term/2" class="tag level11"> </a>
```

Для комбинации тегов

```html
<style>
  .btn {
    /* для всех кнопок */
  }
  .delete {
    /* дополнительные стили для кнопки удалить */
  }
  .add {
    /*  */
  }
  .edit {
    /*  */
  }
</style>
<button class="btn delete">Удалить</button>
<button class="btn add">Добавить</button>
<button class="btn edit">редактировать</button>
```

## Без пробельный селектор

на смешивание двух классов

```html
<html>
  <head>
  <meta charset=utf-8>
  <title>Камни</title>
  <style>
    /* смешанный селектор */
    table.jewel {
      width: 100%
      border: 1px solid #666;
    }
    th {
      background: #009384;
      color: #fff
      text-align: left;
    }
    tr.odd {
      background: #ebd3d7;
    }
    </style>
  </head>
  <body>
    <!-- применится table.jewel -->
  <table class=jewel>
    <tr>
      <!-- th -->
      <th>Название</th>
    </tr>
    <!-- tr.odd -->
    <tr class=odd>
       <td>Алмаз</td>
    </tr>
</html>
```

!!!Разница с псевдоклассами

```scss
article :first-child {
  // элементы-потомки элемента article
}

article:first-child {
  //выберет любой элемент <article>, являющийся первым дочерним элементом другого элемента
}
```

для классов

```scss
.notebox.danger {
  //
}
```

```html
<div class="notebox danger">This note shows danger!</div>
```

## Дочерний Комбинатор > (знак больше)

выбирает только на первом уровне вложенности

```scss
span {
  background-color: white;
}
div > span {
  background-color: DodgerBlue;
}
```

```html
<div>
  <span
    >Span 1 в div (окрасится)
    <span>Span 2 в span, который в div (не окрасится, не прямой потомок)</span>
  </span>
</div>
<span>Span 3. Не в div вообще</span>
```

## Комбинатор всех соседних ~ тильда

Общий комбинатор смежных селекторов (~) разделяет два селектора и находит второй элемент только если ему предшествует первый, и они оба имеют общего родителя

```scss
p ~ span {
  color: red;
}
```

```html
<span>Это не красный.</span>
<p>Здесь параграф.</p>
<code>Тут какой-то код.</code>
<span
  >А здесь span(применится, так как это первый span который идет сразу после
  p)</span
>
```

## Соседние селекторы + (знак плюс)

выберет непосредственно следующего соседа, с которым он имеет одного родителя

```html
<style>
  b + i {
    /* // все что внутри контейнера I следующего после B будет окрашено в красный цвет */
    color: red;
  }
</style>

<head>
  <meta charset="utf-8" />
  <title>Изменение стиля абзаца</title>
  <style>
    /* //для выделения замечаний */
    H2.sic {
    /* …. */
    }
    H2.sic + P {
    /* … */
    }
  </style>
  <!-- … -->
  <body>
    <h1>Заголовок без стиля</h1>
    <h2>Обычный H2</h2>
    <p>
      …текст без стиля…<p>
        <h2 class="sic">
          …Подобзац со стилем
          <h2>
            <!-- так как тег p идет после H2.sic  -->
            <p>…текст со стилем H2.sic + P</p>
          </h2>
        </h2></p>
      >
    </p>
  </body>
</head>
```

## namespace - селектор namespace|selector

```html
<p>This paragraph <a href="#">has a link</a>.</p>

<svg width="400" viewBox="0 0 400 20">
  <a href="#">
    <text x="0" y="15">Link created in SVG</text>
  </a>
</svg>
```

```scss
@namespace svgNamespace url("http://www.w3.org/2000/svg");
@namespace htmlNameSpace url("http://www.w3.org/1999/xhtml");
/* All `<a>`s in the default namespace, in this case, all `<a>`s */
a {
  font-size: 1.4rem;
}
/* no namespace */
|a {
  text-decoration: wavy overline lime;
  font-weight: bold;
}
/* all namespaces (including no namespace) */
*|a {
  color: red;
  fill: red;
  font-style: italic;
}
/* only the svgNamespace namespace, which is <svg> content */
svgNamespace|a {
  color: green;
  fill: green;
}
/* The htmlNameSpace namespace, which is the HTML document */
htmlNameSpace|a {
  text-decoration-line: line-through;
}
```

# идентификаторы

Идентификатор или ID-селектор определяет уникальное имя элемента

```html
<style>
  #help {
  }
</style>
<div id="help">…контент…</div>

Идентификаторы можно применять к тегу

<style>
  p {
  }
  P#opa {
  }
</style>
<p>…контент…</p>
<p id="opa">…контент…</p>
```
