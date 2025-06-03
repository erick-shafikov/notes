# v-if

позволяет отображать

```html
<h1 v-if="shouldDisplay">Conditional content</h1>
```

# v-else

элемент с этой директивой должен идти сразу после v-if или v-else-if

```html
<h1 v-if="awesome">Если awesome === true</h1>
<h1 v-else>Если awesome !== true</h1>
```

# v-else-if

```html
<div v-if="type === 'A'">A</div>
<div v-else-if="type === 'B'">B</div>
<div v-else-if="type === 'C'">C</div>
<div v-else>Not A/B/C</div>
```

# применим к template

```html
<template v-if="ok">
  <h1>Title</h1>
  <p>Paragraph 1</p>
  <p>Paragraph 2</p>
</template>
```

# v-show

Скрывает элемент с помощью css

Разница с v-if

- v-if - это условный рендеринг
- v-if не будет рендерится изначально в DOM если false
- v-show - для частых переключений

# v-if с v-for

Не рекомендуется использовать и на одном элементе из-за неявного приоритета.
