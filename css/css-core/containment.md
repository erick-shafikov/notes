<!-- Изолирование --------------------------------------------------------------------------------------------------------------------------->

# CSS - изолирование

Позволяет изолировать блок в dom, что позволяет оптимизировать работу браузера. Это свойства помогаю решить браузеру
находятся ли элементы в зоне видимости. Начинает отображать элемент 50% от области просмотра

- [contain - свойство которое позволяет определить тип изолирования](./css-props.md/#contain)
- [container = container-name + container-type краткая запись для свойств:](./css-props.md/#container)
- - [container-name позволяет задать имя контейнера](./css-props.md/#container-name)
- - [container-type определяет как вычислять размер контейнера](./css-props.md/#container-type)
- [content-visibility позволяет контролировать видимость содержимого контейнера](./css-props.md/#content-visibility)
- [@container правило, которое активирует функциональность контейнеров](./at-rules.md/#container)

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
