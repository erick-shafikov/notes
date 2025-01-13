Позволяет изолировать блок в dom, что позволяет оптимизировать работу браузера. Это свойства помогаю решить браузеру
находятся ли элементы в зоне видимости. Начинает отображать элемент 50% от области просмотра

# contain

Существует четыре типа ограничения CSS: размер, макет, стиль и краска, которые устанавливаются в контейнере

Примечание: использование layout значений paint, strict или content для этого свойства создает:

Новый содержащий блок (для потомков, position свойство которых равно absolute или fixed).
Новый контекст наложения
Новый контекст форматирования блока

- size - Размер элемента может быть вычислен изолированно, без учета дочерних элементов
- inline-size - К элементу применяется ограничение размера в строке
- layout - Внутренняя компоновка элемента изолирована от остальной части страницы. Это означает, что ничто извне элемента не влияет на его внутреннюю компоновку, и наоборот.
- style - Счетчики и кавычки ограничены элементом и его содержимым.
- paint - Потомки элемента не отображаются за его пределами.

```scss
 {
  contain: none; //Элемент отображается как обычно, без применения сдерживания.
  contain: strict; //Все правила сдерживания применяются к элементу: size, layout, paint, style
  contain: content; //блок независимый, невидимые не будет отрисовать: layout paint style
  contain: size; // размер элемента может быть вычислен изолировано, не работает в паре с contain-intrinsic-size
  contain: inline-size; // строчное
  contain: layout; // Внутренняя компоновка элемента изолирована от остальной части страницы
  //что ничто извне элемента не влияет на его внутреннюю компоновку,
  contain: style; //Для свойств, которые могут влиять не только на элемент и его потомков, эффекты не выходят за пределы содержащего элемента
  contain: paint; //Потомки элемента не отображаются за его пределами.
}
```

- В некоторых случаях, особенно при использовании строгого значения strict, браузер может потребовать дополнительных ресурсов для оптимизации рендеринга. Поэтому важно тестировать и измерять производительность при использовании свойства.contain применяется к самому элементу и его содержимому, но не влияет на элементы, вложенные внутри него. Если требуется оптимизировать взаимодействие внутри вложенных элементов, нужно применить свойство contain к каждому из них отдельно.
- Свойство наиболее полезно в ситуациях, когда у вас есть небольшой набор элементов, которые могут быть легко изолированы и оптимизированы.
- В случае сложных макетов с большим количеством элементов, использовать contain бывает сложно и неэффективно

<!-- container ----------------------------------------------------------------------------------------------------------------------------->

# container:

container = container-name + container-type

```scss
.container {
  container: my-layout;
  container: my-layout / size;
}
```

## container-name

Определяет имя контейнера, не должно быть or, and, not, или default.

```scss
 {
  container-name: myLayout;
  container-name: myPageLayout myComponentLibrary; //несколько имен
}
```

## container-type

```scss
 {
  container-type: normal;
  container-type: size; //по inline и block модели
  container-type: inline-size; //по строчной
}
```

# content-visibility

Позволяет сделать содержимое контейнера невидимым. Основное применение для создание плавных анимаций, при которых контент плавно пропадает.
В анимации нужно включить transition-behavior: content-visibility

```scss
 {
  content-visibility: visible; //обычное отображение элемента
  content-visibility: hidden; // не будет доступно для поиска, фокусировки
  content-visibility: auto; //contain: content
}
```

Второе применение экономия ресурсов при рендеринге

# @container

Позволяет настроить медиа запросы относительно элемента, а не vp. Задает контекст ограничения

```scss
// настройка для контейнера относительного которого будут производится измерения
.container {
  container-type: size; //size - измеряет как как inline или блочный
  container-type: inline-size; //inline-size - как inline
  container-type: normal; //Отключает запросы на размеры контейнера
}
```

```html
<!-- контейнер -->
<div class="post">
  <div class="card">
    <h2>Card title</h2>
    <p>Card content</p>
  </div>
</div>
```

```scss
.post {
  container-type: inline-size;
}
// контент .card и h2 будет изменять font-size при min-width: 700px
@container (min-width: 700px) {
  .card h2 {
    font-size: 2em;
  }
}
```

для множественных контейнеров, именованные контейнеры

```scss
.post {
  container-type: inline-size;
  // название контейнера
  container-name: sidebar;
  // сокращенное свойство
  container: sidebar / inline-size;
}
// использование
@container sidebar (min-width: 700px) {
  .card {
    font-size: 2em;
  }
}
```

доступны новые единицы измерения

- cqw: 1% от ширины контейнера
- cqh: 1% от высоты
- cqi: 1% от inline размера
- cqb: 1% от блочного размера
- cqmin: минимум от cqi и cqb
- cqmax: максимум от cqi и cqb

```scss
// использование
@container (min-width: 700px) {
  .card h2 {
    font-size: max(1.5em, 1.23em + 2cqi);
  }
}
```

Доступные свойства для условия: aspect-ratio, block-size, height, inline-size, orientation, width

вложенные

```scss
@container summary (min-width: 400px) {
  @container (min-width: 800px) {
    /* <stylesheet> */
  }
}
```

## Container style queries

c помощью функции style можно ссылать на стиль контейнера

```scss
@container style(<style-feature>),
    not style(<style-feature>),
    style(<style-feature>) and style(<style-feature>),
    style(<style-feature>) or style(<style-feature>) {
  /* <stylesheet> */
}

// пример

@container style(--themeBackground),
    not style(background-color: red),
    style(color: green) and style(background-color: transparent),
    style(--themeColor: blue) or style(--themeColor: purple) {
  /* <stylesheet> */

  //--themeColor: blue - незарегистрированное пользовательское свойство
}
```

Пример использования

```html
<div class="card">
  <!-- 1 -->
  <div class="post-meta">
    <h2>Card title</h2>
    <p>My post details.</p>
  </div>
  <!-- 2 -->
  <div class="post-excerpt">
    <p>
      A preview of my <a href="https://example.com">blog post</a> about cats.
    </p>
  </div>
</div>
```

```scss
// первый контейнер безымянный
.post-meta {
  container-type: inline-size;
}

// второй контейнер excerpt
.post-excerpt {
  container-type: inline-size;
  container-name: excerpt;
}

// будет применен для контейнера excerpt, для тега p
@container excerpt (min-width: 400px) {
  p {
    visibility: hidden;
  }
}

// будет применен для безымянного контейнера, для тега p
@container (min-width: 400px) {
  p {
    font-size: 2rem;
  }
}
```
