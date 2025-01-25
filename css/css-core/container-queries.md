# container:

container = container-name + container-type

```scss
.container {
  container: my-layout;
  container: my-layout / size;
}
```

Пример

```html
<!-- post - контейнер (sidebar) -->
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
  container-name: sidebar;
}

@container sidebar (min-width: 400px) {
  /* <stylesheet> */
}
```

## container-name

Определяет имя контейнера, не должно быть or, and, not, или default.

```scss
.container-name {
  container-name: myLayout;
  container-name: myPageLayout myComponentLibrary; //несколько имен
}
```

## container-type

```scss
.container-type {
  container-type: normal;
  container-type: size; //по inline и block модели
  container-type: inline-size; //по строчной
}
```

# content-visibility

Позволяет сделать содержимое контейнера невидимым. Основное применение для создание плавных анимаций, при которых контент плавно пропадает.
В анимации нужно включить transition-behavior: content-visibility

```scss
.content-visibility {
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
