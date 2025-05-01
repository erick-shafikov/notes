window.document - входная точка в dom

#

## конструктор

## статические свойства

## статические методы

## свойства экземпляра

- activeElement ⇒ элемент на котором фокус
- adoptedStyleSheets - для использования с CSSStyleSheet

```js
// Create an empty "constructed" stylesheet
const sheet = new CSSStyleSheet();
// Apply a rule to the sheet
sheet.replaceSync("a { color: red; }");

// Apply the stylesheet to a document
document.adoptedStyleSheets = [sheet];
```

- alinkColor (Устарело) - цвет ссылок
- all (Устарело) ⇒ все элементы
- anchors (Устарело) ⇒ все якоря (а)
- applets (Устарело)
- bgColor (Устарело)
- body ⇒ узел body или frameSet
- characterSet ⇒ кодировка
- childElementCount ⇒ количество дочерних эл-тов
- children ⇒ живую коллекцию из дочерних элементов
- compatMode (нестандартная возможность)
- contentType
- cookie - геттер/сеттер для кук
- currentScript ⇒ скрипт который выполняется
- defaultView ⇒ window или null
- designMode ⇒ "on" и "off" режим правки документа
- dir ⇒ 'rtl' или 'ltr'
- doctype ⇒ текущий адрес документа
- documentElement ⇒ html элемент
- documentURI
- domain (Устарело) - получает/устанавливает доменную часть
- embeds ⇒ список embed
- featurePolicy - Экспериментальная - возможность
- fgColor (Устарело)
- firstElementChild
- fonts ⇒ FontFaceSet
- forms ⇒ HTMLCollection - который содержит все формы
- fragmentDirective ⇒ FragmentDirective
- fullscreen (Устарело) ⇒ boolean
- fullscreenElement (-sf) ⇒ элемент который в fullscreen режиме
- fullscreenEnabled (-sf) ⇒
- head ⇒ head элемент
- hidden ⇒ скрыта ли страница
- images ⇒ коллекцию всех документов (readonly)
- implementation
- lastElementChild
- lastModified ⇒ дата изменения документа
- lastStyleSheetSetНе стандартно (Устарело)
- linkColor (Устарело)
- links ⇒ список все area и a
- location ⇒ объект Location (readonly)
- pictureInPictureElement (-ff) ⇒ элемент который в режиме pictureInPicture
- pictureInPictureEnabled (-ff) -
- plugins ⇒ коллекция embed (readonly)
- pointerLockElement (-sf)
- preferredStyleSheetSetНе стандартно (Устарело)
- prerendering Экспериментальная возможность ⇒ readonly (readonly)
- readyState ⇒ loading | interactive | complete
- referrer ⇒ URI страницы
- rootElement (Устарело)
- scripts ⇒ HTMLCollection скриптов в документе
- scrollingElement ⇒ Element прокрутки документа (readonly)
- selectedStyleSheetSetНе стандартно (Устарело)
- styleSheets ⇒ StyleSheetList
- styleSheetSetsНе стандартно (Устарело)
- timeline ⇒ для DocumentTimeline.
- title ⇒ титул документа
- URL ⇒ строку URL документа HTML.
- visibilityState ⇒ 'visible' | 'hidden' если страница в фоне или свернута (readonly)
- vlinkColor (Устарело)
- xmlEncoding (Устарело)
- xmlVersion (Устарело)

## методы экземпляра

## события

## обработчики события

⇒
