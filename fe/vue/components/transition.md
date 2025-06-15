!!! transition может анимировать только один вложенный элемент

- vue определяет есть ли элементы с аномалиями
- если есть, то прослушивает хуки
- если нет, то удаление вставка будет в следующем кадре

Последовательность классов:

- v-enter-from - добавляется перед вставкой элемента, удаляется на следующий кадр
- v-enter-active - в процессе вставки
- v-enter-to - появление элемента
- v-leave-from - старт исчезновения
- v-leave-active - активная стадия исчезновения
- v-leave-to - предпоследний кадр перед исчезновением

Параметр name - определяет анимацию

```vue
<template>
  <!-- fade - класс с анимацией, в параметре name -->
  <transition name="fade">
    <div class="alert alert-success" v-show="showAlert">
      <p>Lorem</p>
    </div>
  </transition>
</template>
<script></script>
<style module>
/* стартовая позиция */
/* к fade добавятся классы -enter-from*/
.fade-enter-from {
  opacity: 0;
}
/* финальная */
/* к fade добавятся классы -enter-active*/
.fade-enter-active {
  transition: opacity 0.5s;
}
/* при пропадании */
/* к fade добавятся классы -leave-active*/
.fade-leave-active {
  transition: opacity 0.5s;
  opacity: 0;
}
</style>
```

с помощью keyframes

```vue
<template></template>
<script></script>
<style>
.sample {
  overflow: hidden;
}

.fade-enter-active {
  animation: fadeIn 0.5s;
}

.fade-leave-active {
  animation: fadeOut 0.5s;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateX(-100px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes fadeOut {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100px);
  }
}
</style>
```

# classnames

Список классов:

- enter-from-class
- enter-active-class
- enter-to-class
- leave-from-class
- leave-active-class
- leave-to-class

```vue
<template>
  <!-- animate__animated animate__tada и другие классы определены в стиле -->
  <transition
    enter-active-class="animate__animated animate__tada"
    leave-active-class="animate__animated animate__fadeOutLeft"
  >
    <div class="alert alert-success" v-show="showAlert">
      <p>Lorem</p>
    </div>
  </transition>
</template>
<script></script>
```

# :duration

```vue
<template>
  <Transition :duration="{ enter: 500, leave: 800 }">...</Transition>
</template>
<script></script>
```

# :type

```vue
<template>
  <Transition type="animation">...</Transition>
</template>
<script></script>
```

# :css

если анимация происходит с помощью хуков на JS

```vue
<template><Transition ... :css="false"> ... </Transition></template>
<script></script>
```

# appear

анимация на появление

```vue
<template><Transition appear> ... </Transition></template>
<script></script>
```

# mode

так как анимации не проходят с position absolute избежать сдвиг

```vue
<template>
  <Transition mode="out-in"> ... </Transition>
</template>
<script></script>
```

# хуки

```vue
<template>
  <Transition
    @before-enter="onBeforeEnter"
    @enter="onEnter"
    @after-enter="onAfterEnter"
    @enter-cancelled="onEnterCancelled"
    @before-leave="onBeforeLeave"
    @leave="onLeave"
    @after-leave="onAfterLeave"
    @leave-cancelled="onLeaveCancelled"
  >
    <!-- ... -->
  </Transition>
</template>
<script>
// вызывается перед вставкой элемента в DOM.
// используется для установки состояния "enter-from" элемента
function onBeforeEnter(el) {}

// вызывается через один кадр после вставки элемента.
// используйте его для запуска анимации входа.
function onEnter(el, done) {
  // вызов обратного вызова done для индикации окончания перехода
  // необязателен, если используется в сочетании с CSS
  done();
}

// вызывается по завершении перехода enter.
function onAfterEnter(el) {}

// вызывается, когда переход enter отменяется до завершения.
function onEnterCancelled(el) {}

// вызывается перед хуком leave.
// В большинстве случаев следует использовать только хук leave
function onBeforeLeave(el) {}

// вызывается, когда начинается переход к leave.
// используйте его для запуска анимации ухода.
function onLeave(el, done) {
  // вызов обратного вызова done для индикации окончания перехода
  // необязательно, если используется в сочетании с CSS
  done();
}

// вызывается, когда переход к leave завершен
// элемент был удален из DOM.
function onAfterLeave(el) {}

// доступно только при использовании переходов v-show
function onLeaveCancelled(el) {}
</script>
```

# Переисользуемые

```vue
<!-- MyTransition.vue -->

<template>
  <!-- обернуть встроенный компонент Transition -->
  <Transition name="my-transition" @enter="onEnter" @leave="onLeave">
    <slot></slot>
    <!-- передать содержимое слота -->
  </Transition>
</template>

<style>
/*
  Необходимый CSS...
  Примечание: избегайте использования здесь <style scoped>,
  так как это не относится к содержимому слота.
*/
</style>
```

```vue
<template>
  <MyTransition>
    <div v-if="show">привет</div>
  </MyTransition>
</template>
<script></script>
```

<!-- ------------------------- -->

```vue
<template></template>
<script></script>
```
