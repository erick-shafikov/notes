window.document - входная точка в dom

#

## статические методы

позволяет из строки сделать Document

- parseHTMLUnsafe(html) ⇒ Document. html - строка,

```js
Document.parseHTMLUnsafe(input);
```

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

- adoptNode() - позволяет вставить кусок из одного документа в другой

```js
const iframe = document.querySelector("iframe");
const iframeImages = iframe.contentDocument.querySelectorAll("img");
const newParent = document.getElementById("images");

iframeImages.forEach((imgEl) => {
  newParent.appendChild(document.adoptNode(imgEl));
});
```

- append(param1, param2, paramN) - вставит что-то в конец документа, где param1, param2, paramN - узлы для вставки

- browsingTopics() - темы (Experimental Non-standard)
- caretPositionFromPoint(x, y, options) ⇒ CaretPosition объект
- caretRangeFromPoint() - позиция курсора в тексте (Non-standard)
- clear() - Deprecated
- close() - закроет запись в документ

```js
// открытие документа для записи в него.
// запись содержимого документа.
// закрытие документа.
document.open();
document.write("<p>The one and only content.</p>");
document.close();
```

- createAttribute() - создает атрибут дял использования в тегах

```js
const node = document.getElementById("div1");
const a = document.createAttribute("my_attrib");
a.value = "newVal";
node.setAttributeNode(a);
console.log(node.getAttribute("my_attrib")); // "newVal"
```

- createAttributeNS() - createAttribute + NS
- createCDATASection() - создает CDATA узел
- createComment() - создаст коммент
- createDocumentFragment() - интерфейс для создания фрагментов
- createElement(tagName, {is:'user-element-name'}) ⇒ элемент
- createElementNS()
- createEvent() - Deprecated
- createExpression() - дял XPathExpression
- createNodeIterator(root, whatToShow, filter)
- - root - куда встраивать
- - whatToShow - 0xFFFFFFFF по умолчанию
- - filter - функция
- createNSResolver() - Deprecated
- createProcessingInstruction() - для XML

- createRange(startNode, startOffset) ⇒ Range.
- createTextNode(data) - создаст текстовый узел
- - data - текстовый контент
- createTouch() - Non-standard Deprecated
- createTouchList() - Non-standard Deprecated
- createTreeWalker() ⇒ TreeWalker
- elementFromPoint(x, y) ⇒ Element самый верхний элемент
- elementsFromPoint() ⇒ Element[] от самого верхнего до самого нижнего
- enableStyleSheetsForSet() - Non-standard Deprecated
- evaluate() ⇒ XPathResult
- execCommand() - Deprecated для работы с документов в режиме редактирования

- exitFullscreen() - для выхода из полноэкранного режима
- exitPictureInPicture() - для выхода из PictureInPicture режима
- exitPointerLock() -
- getAnimations() ⇒ Animation[] со всеми анимаций
- getElementById() ⇒ Element по id, регистр зависимый
- getElementsByClassName() ⇒ Element[] с указанным классом
- getElementsByName() ⇒ Element[] с указанным атрибутом name
- getElementsByTagName() ⇒ Element[] по тегу
- getElementsByTagNameNS() - с учетом NS
- getSelection() => Selection в котором содержится информация о выделенном тексте
- hasFocus() - имеет ли элемент или вложенные фокус
- hasStorageAccess()⇒Promise(Boolean) есть ли доступ к cookie
- hasUnpartitionedCookieAccess() ⇒ Promise(Boolean) есть ли доступ к сторонними cookie
- importNode() - создаст копию узла

```js
var iframe = document.querySelector("iframe");
var oldNode = iframe.contentWindow.document.getElementById("myNode");
var newNode = document.importNode(oldNode, true);
document.getElementById("container").appendChild(newNode);
```

- moveBefore(movedNode, referenceNode) - двигает movedNode, referenceNode - перед чем вставить
- mozSetImageElement(imageElementId, imageElement) - Non-standard - меняет фон
- open() - откроет возможность редактировать элемент

```js
//первое использование
document.open();
document.write("<p>Hello world!</p>");
document.write("<p>I am a fish</p>");
document.write("<p>The number is 42</p>");
document.close();

//второе открытие ссылок
document.open("https://www.github.com", "", "noopener=true");
```

- prepend(param1, param2, paramN) - вставит перед документом
- queryCommandEnabled() - Non-standard,Deprecated
- queryCommandState() - Non-standard, Deprecated
- queryCommandSupported() - Non-standard Deprecated
- querySelector(selectors) => первый Element в соответствии с селектором
- querySelectorAll() => нединамически NodeList
- releaseCapture() - Non-standard
- replaceChildren(param1, param2) - для замены дочерних элементов
- requestStorageAccess({all:false, cookies, sessionStorage, localStorage, indexedDB}) - запрос для сторонних куки файлов
- requestStorageAccessFor(requestedOrigin) - Experimental - установит разрешение
  -startViewTransition(updateCallback) => ViewTransition для SPA контролировать переходы между страницами, updateCallback вызывается после перехода
- write() - Deprecated - открывает документ для редактирования
- writeln() - Выводит в документ строку со знаком перевода каретки в конце.

## события

- afterscriptexecute - Non-standard Deprecated - окончание работы скрипта
- beforescriptexecute - Non-standard Deprecated - старт работы скрипта
- copy - ClipboardEvent - при копировании
- cut
- DOMContentLoaded - документ загружен без ожидания стилей, изображений, фреймов
- fullscreenchange - переход в fullscreen режим
- fullscreenerror - если браузер не умеет в fullscreen
- paste - при вставки
- pointerlockchange - заблокирован ли указатель
- pointerlockerror
- prerenderingchangeExperimental - запускается для предварительно отрисованного документа
- readystatechange - при изменении readyState статуса документа
- scroll - при прокрутке страницы

```js
// Источник: http://www.html5rocks.com/en/tutorials/speed/animations/

let last_known_scroll_position = 0;
let ticking = false;

function doSomething(scroll_pos) {
  // Делаем что-нибудь с позицией скролла
}

window.addEventListener("scroll", function (e) {
  last_known_scroll_position = window.scrollY;

  if (!ticking) {
    window.requestAnimationFrame(function () {
      doSomething(last_known_scroll_position);
      ticking = false;
    });

    ticking = true;
  }
});
```

- scrollend - документ пролистан
- scrollsnapchange - Experimental - прокручен контейнер
- scrollsnapchanging- Experimental
- securitypolicyviolation - при нарушении CSP
- selectionchange - при изменении выделения
- visibilitychange - при смене видимости вкладки
