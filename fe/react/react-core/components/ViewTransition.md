# ViewTransition

РђРЅРёРјРёСЂСѓРµС‚ СЌР»РµРјРµРЅС‚С‹ РїСЂРё РїРѕСЏРІР»РµРЅРёРё, РёСЃС‡РµР·РЅРѕРІРµРЅРёРё, РїРµСЂРµРјРµС‰РµРЅРёРё Рё РёР·РјРµРЅРµРЅРёРё СЂР°Р·РјРµСЂР° С‡РµСЂРµР· [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/View_Transition_API). РЎС‚Р°Р±РёР»РµРЅ РІ React 19.3.

РђРЅРёРјР°С†РёРё СЃСЂР°Р±Р°С‚С‹РІР°СЋС‚ С‚РѕР»СЊРєРѕ РґР»СЏ РѕР±РЅРѕРІР»РµРЅРёР№, РїРѕРјРµС‡РµРЅРЅС‹С… РєР°Рє Transition: `startTransition`, reveal РІ `<Suspense>` РёР»Рё РѕР±РЅРѕРІР»РµРЅРёСЏ РёР· `useDeferredValue`. РћР±С‹С‡РЅС‹Рµ СЃСЂРѕС‡РЅС‹Рµ РѕР±РЅРѕРІР»РµРЅРёСЏ Р°РЅРёРјР°С†РёРё РЅРµ Р·Р°РїСѓСЃРєР°СЋС‚.

**РўРёРїС‹ Р°РЅРёРјР°С†РёР№ РїРѕ РёР·РјРµРЅРµРЅРёСЋ РґРµСЂРµРІР°:**

- **enter** вЂ” `<ViewTransition>` РґРѕР±Р°РІР»РµРЅ РІ РґРµСЂРµРІРѕ
- **exit** вЂ” `<ViewTransition>` СѓРґР°Р»С‘РЅ РёР· РґРµСЂРµРІР°
- **update** вЂ” РґРµС‚Рё РјРµРЅСЏСЋС‚ СЃС‚РёР»СЊ РёР»Рё СЃРѕРґРµСЂР¶РёРјРѕРµ
- **share** вЂ” РёРјРµРЅРѕРІР°РЅРЅС‹Р№ `<ViewTransition>` СѓРґР°Р»С‘РЅ РІ РѕРґРЅРѕРј РјРµСЃС‚Рµ Рё РґРѕР±Р°РІР»РµРЅ РІ РґСЂСѓРіРѕРј

## Props

| Prop | РўРёРї | РћРїРёСЃР°РЅРёРµ |
|---|---|---|
| `enter` | `string \| object` | РђРЅРёРјР°С†РёСЏ РїСЂРё РґРѕР±Р°РІР»РµРЅРёРё. РЎС‚СЂРѕРєР° вЂ” РёРјСЏ CSS-РєР»Р°СЃСЃР°; РѕР±СЉРµРєС‚ вЂ” РјР°РїРїРёРЅРі С‚РёРїР° РїРµСЂРµС…РѕРґР° РЅР° РєР»Р°СЃСЃ: `{'next': 'from-right', 'previous': 'from-left'}` |
| `exit` | `string \| object` | РђРЅРёРјР°С†РёСЏ РїСЂРё СѓРґР°Р»РµРЅРёРё. РђРЅР°Р»РѕРіРёС‡РЅРѕ `enter` |
| `update` | `string \| "auto"` | РђРЅРёРјР°С†РёСЏ РїСЂРё РѕР±РЅРѕРІР»РµРЅРёРё СЃРѕРґРµСЂР¶РёРјРѕРіРѕ. `"auto"` вЂ” Р°РЅРёРјРёСЂРѕРІР°С‚СЊ РїРµСЂРµС…РѕРґ РјРµР¶РґСѓ fallback Рё РєРѕРЅС‚РµРЅС‚РѕРј РІ `<Suspense>` |
| `default` | `string \| "none"` | РђРЅРёРјР°С†РёСЏ РґР»СЏ С‚РёРїРѕРІ, РЅРµ РїРѕРєСЂС‹С‚С‹С… РґСЂСѓРіРёРјРё РїСЂРѕРїСЃР°РјРё. `"none"` вЂ” РѕС‚РєР»СЋС‡РёС‚СЊ Р°РЅРёРјР°С†РёСЋ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ |
| `name` | `string` | РРјСЏ РґР»СЏ shared-СЌР»РµРјРµРЅС‚РѕРІ (РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ СѓРЅРёРєР°Р»СЊРЅС‹Рј РІРѕ РІСЃС‘Рј РїСЂРёР»РѕР¶РµРЅРёРё РІ РєР°Р¶РґС‹Р№ РјРѕРјРµРЅС‚) |
| `key` | `string` | РџСЂРё СЃРјРµРЅРµ РєР»СЋС‡Р° СЃСЂР°Р±Р°С‚С‹РІР°РµС‚ РїРµСЂРµС…РѕРґ (exit СЃС‚Р°СЂРѕРіРѕ + enter РЅРѕРІРѕРіРѕ) |
| `onEnter` | `(instance) => void` | РЎРѕР±С‹С‚РёРµ РґР»СЏ РёРјРїРµСЂР°С‚РёРІРЅРѕР№ Р°РЅРёРјР°С†РёРё С‡РµСЂРµР· Web Animations API РїСЂРё enter |
| `onExit` | `(instance) => void` | РЎРѕР±С‹С‚РёРµ РїСЂРё exit |
| `onUpdate` | `(instance) => void` | РЎРѕР±С‹С‚РёРµ РїСЂРё update |
| `onShare` | `(instance) => void` | РЎРѕР±С‹С‚РёРµ РїСЂРё shared-РїРµСЂРµС…РѕРґРµ |

## РџСЂРёРјРµСЂ: enter/exit

```tsx
import { ViewTransition, useState, startTransition } from 'react';

export default function Component(): JSX.Element {
  const [showItem, setShowItem] = useState<boolean>(false);

  return (
    <>
      <button
        onClick={() => {
          startTransition(() => {
            setShowItem((prev) => !prev);
          });
        }}>
        {showItem ? 'вћ–' : 'вћ•'}
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

## РџСЂРёРјРµСЂ: РєР°СЂСѓСЃРµР»СЊ СЃ `addTransitionType`

Р Р°Р·РЅС‹Рµ Р°РЅРёРјР°С†РёРё РІ Р·Р°РІРёСЃРёРјРѕСЃС‚Рё РѕС‚ РЅР°РїСЂР°РІР»РµРЅРёСЏ вЂ” РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ `addTransitionType` Рё РѕР±СЉРµРєС‚РЅР°СЏ С„РѕСЂРјР° РїСЂРѕРїСЃРѕРІ `enter`/`exit`.

```tsx
import {
  ViewTransition,
  addTransitionType,
  useState,
  startTransition,
} from 'react';
import './animations.css';

export default function Carousel(): JSX.Element {
  const [selected, setSelected] = useState<number>(0);
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
        <button onClick={previousSlide}>в¬…пёЏ</button>
        <button onClick={nextSlide}>вћЎпёЏ</button>
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

## РЎ Suspense (`update="auto" default="none"`)

```tsx
<ViewTransition update="auto" default="none">
  <Suspense fallback={<VideoPlaceholder />}>
    <LazyVideo />
  </Suspense>
</ViewTransition>
```

`update="auto"` Р°РЅРёРјРёСЂСѓРµС‚ РїРµСЂРµС…РѕРґ РјРµР¶РґСѓ fallback Рё Р·Р°РіСЂСѓР¶РµРЅРЅС‹Рј РєРѕРЅС‚РµРЅС‚РѕРј; `default="none"` РѕС‚РєР»СЋС‡Р°РµС‚ РѕСЃС‚Р°Р»СЊРЅС‹Рµ Р°РЅРёРјР°С†РёРё.
