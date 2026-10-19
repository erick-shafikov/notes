# ViewTransition

Анимирует элементы при появлении, исчезновении, перемещении и изменении размера через [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API). Стабилен в React 19.3.

Анимации срабатывают только для обновлений, помеченных как Transition: `startTransition`, reveal в `<Suspense>` или обновления из `useDeferredValue`. Обычные срочные обновления анимации не запускают.

**Типы анимаций по изменению дерева:**

- **enter** — `<ViewTransition>` добавлен в дерево
- **exit** — `<ViewTransition>` удалён из дерева
- **update** — дети меняют стиль или содержимое
- **share** — именованный `<ViewTransition>` удалён в одном месте и добавлен в другом

## Props

| Prop | Тип | Описание |
|---|---|---|
| `enter` | `string \| object` | Анимация при добавлении. Строка — имя CSS-класса; объект — маппинг типа перехода на класс: `{'next': 'from-right', 'previous': 'from-left'}` |
| `exit` | `string \| object` | Анимация при удалении. Аналогично `enter` |
| `update` | `string \| "auto"` | Анимация при обновлении содержимого. `"auto"` — анимировать переход между fallback и контентом в `<Suspense>` |
| `default` | `string \| "none"` | Анимация для типов, не покрытых другими пропсами. `"none"` — отключить анимацию по умолчанию |
| `name` | `string` | Имя для shared-элементов (должно быть уникальным во всём приложении в каждый момент) |
| `key` | `string` | При смене ключа срабатывает переход (exit старого + enter нового) |
| `onEnter` | `(instance) => void` | Событие для императивной анимации через Web Animations API при enter |
| `onExit` | `(instance) => void` | Событие при exit |
| `onUpdate` | `(instance) => void` | Событие при update |
| `onShare` | `(instance) => void` | Событие при shared-переходе |

## Пример: enter/exit

```jsx
import { ViewTransition, useState, startTransition } from 'react';

export default function Component() {
  const [showItem, setShowItem] = useState(false);

  return (
    <>
      <button
        onClick={() => {
          startTransition(() => {
            setShowItem((prev) => !prev);
          });
        }}>
        {showItem ? '➖' : '➕'}
      </button>

      {showItem && (
        <ViewTransition>
          <Video video={videos[0]} />
        </ViewTransition>
      )}
    </>
  );
}
```

## Пример: карусель с `addTransitionType`

Разные анимации в зависимости от направления — используется `addTransitionType` и объектная форма пропсов `enter`/`exit`.

```jsx
import {
  ViewTransition,
  addTransitionType,
  useState,
  startTransition,
} from 'react';
import './animations.css';

export default function Carousel() {
  const [selected, setSelected] = useState(0);
  const video = videos[selected];

  function nextSlide() {
    startTransition(() => {
      addTransitionType('next');
      setSelected(c => c + 1 < videos.length ? c + 1 : 0);
    });
  }

  function previousSlide() {
    startTransition(() => {
      addTransitionType('previous');
      setSelected(c => c > 0 ? c - 1 : videos.length - 1);
    });
  }

  return (
    <>
      <div className="button-container">
        <button onClick={previousSlide}>⬅️</button>
        <button onClick={nextSlide}>➡️</button>
      </div>

      <ViewTransition
        key={video.id}
        enter={{
          'next': 'from-right',
          'previous': 'from-left',
        }}
        exit={{
          'next': 'to-left',
          'previous': 'to-right',
        }}
      >
        <Video video={video} />
      </ViewTransition>
    </>
  );
}
```

CSS (`animations.css`):

```css
::view-transition-old(*),
::view-transition-new(*) {
  animation-duration: 250ms;
  animation-timing-function: cubic-bezier(0.22, 1, 0.36, 1);
}

::view-transition-new(.from-right) { --offset: 100%; animation-name: slide-in; }
::view-transition-new(.from-left)  { --offset: -100%; animation-name: slide-in; }
::view-transition-old(.to-right)   { --offset: 100%; animation-name: slide-out; }
::view-transition-old(.to-left)    { --offset: -100%; animation-name: slide-out; }

@keyframes slide-in {
  from { transform: translateX(var(--offset)); opacity: 0; }
}

@keyframes slide-out {
  to   { transform: translateX(var(--offset)); opacity: 0; }
}
```

## С Suspense (`update="auto" default="none"`)

```jsx
<ViewTransition update="auto" default="none">
  <Suspense fallback={<VideoPlaceholder />}>
    <LazyVideo />
  </Suspense>
</ViewTransition>
```

`update="auto"` анимирует переход между fallback и загруженным контентом; `default="none"` отключает остальные анимации.
