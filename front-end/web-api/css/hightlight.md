CSS Custom Highlight API

- Создание Rangeобъектов.
- Создание Highlight объектов для этих диапазонов.
- Регистрация основных моментов с использованием HighlightRegistry.
- Стилизация бликов с использованием ::highlight() псевдоэлемента.

```html
<label>Search within text <input id="query" type="text" /></label>
<article>
  <p>Maxime ...</p>
  <p>Maiores ... molestiae tempora? Vitae.</p>
  <p>Dolorum ... q</p>
</article>
```

```css
::highlight(search-results) {
  background-color: #f06;
  color: white;
}
```

```js
const query = document.getElementById("query");
const article = document.querySelector("article");

// проход по всем узлам createTreeWalker
const treeWalker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT);
const allTextNodes = [];
let currentNode = treeWalker.nextNode();
while (currentNode) {
  allTextNodes.push(currentNode);
  currentNode = treeWalker.nextNode();
}

// Listen to the input event to run the search.
query.addEventListener("input", () => {
  //если не поддерживается
  if (!CSS.highlights) {
    article.textContent = "CSS Custom Highlight API not supported.";
    return;
  }

  // сброс предыдущего выделения
  CSS.highlights.clear();

  // обработка строки поиска
  const str = query.value.trim().toLowerCase();
  if (!str) {
    return;
  }

  const ranges = allTextNodes
    .map((el) => {
      //упаковка всех элементов в массив с ссылкой на элемент и текстом в нижнем регистре
      return { el, text: el.textContent.toLowerCase() };
    })
    .map(({ text, el }) => {
      const indices = [];
      let startPos = 0;
      // проход по всем, если есть
      while (startPos < text.length) {
        const index = text.indexOf(str, startPos);
        if (index === -1) break;
        indices.push(index);
        startPos = index + str.length;
      }

      // создается объект range
      return indices.map((index) => {
        const range = new Range();
        range.setStart(el, index);
        range.setEnd(el, index + str.length);
        return range;
      });
    });

  // использование api
  const searchResultsHighlight = new Highlight(...ranges.flat());

  // добавления атрибуты css
  CSS.highlights.set("search-results", searchResultsHighlight);
});
```

# Highlight()

свойства экземпляра:

- priority
- size
- type

методы экземпляра:

- add()
- clear()
- delete()
- entries()
- forEach()
- has()
- keys()
- values()

```js
const parentNode = document.getElementById("foo");

// Create a couple of ranges.
const range1 = new Range();
range1.setStart(parentNode, 10);
range1.setEnd(parentNode, 20);

const range2 = new Range();
range2.setStart(parentNode, 40);
range2.setEnd(parentNode, 60);

// Create a custom highlight for these ranges.
const highlight = new Highlight(range1, range2);

// Register the ranges in the HighlightRegistry.
CSS.highlights.set("my-custom-highlight", highlight);
```

```css
::highlight(my-custom-highlight) {
  background-color: peachpuff;
}
```

# HighlightRegistry

позволяет установить взаимодействие с css
