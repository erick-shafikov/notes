# view-transition-name (-ff)

Анимация срабатывает при переходе от страницы к странице

```scss
.view-transition-name {
  view-transition-name: header;
  view-transition-name: figure-caption;
}
```

# @view-transition (-ff)

```scss
@view-transition {
  navigation: auto;
  navigation: none; // Документ не будет подвергнут переходу вида.
}
```

Активируется с помощью псевдоэлементов

```scss
@view-transition {
  navigation: auto;
}

@keyframes move-out {
  from {
    transform: translateY(0%);
  }

  to {
    transform: translateY(-100%);
  }
}

@keyframes move-in {
  from {
    transform: translateY(100%);
  }

  to {
    transform: translateY(0%);
  }
}

/* Apply the custom animation to the old and new page states */
::view-transition-old(root) {
  animation: 0.4s ease-in both move-out;
}

::view-transition-new(root) {
  animation: 0.4s ease-in both move-in;
}
```

## ::view-transition

подключает переход к корневому элементу

```scss
html::view-transition {
  position: fixed;
  inset: 0;
}
```

## ::view-transition-image-pair (-ff)

## ::view-transition-group() (-ff)

## ::view-transition-new() (-ff)

## ::view-transition-old() (-ff)
