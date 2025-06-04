# разрыв и перенос

# word-break

Где будет установлен перевод на новую строку

```scss
.word-break {
  word-break: normal;
  word-break: break-all;
  word-break: keep-all;
  word-break: break-word;
}
```

# text-wrap

перенос слов

```scss
.text-wrap {
  text-wrap: wrap; //обычный перенос при переполнение
  text-wrap: nowrap; //отмена переноса
  text-wrap: balance; //лучшее соотношение в плане длины строк, автоматически распределяет текст в равных пропорциях (убирает вдовы)
  text-wrap: pretty; // более медленный алгоритм wrap
  text-wrap: stable;
}
```

менее поддерживаемые свойство - text-wrap-mode, text-wrap-style

```scss
.text-wrap {
  text-wrap-mode: wrap | nowrap;
  text-wrap-style: auto;
  text-wrap-style: balance; ||
  text-wrap-style: pretty;
  text-wrap-style: stable;
}
```

# overflow-wrap

разрыв сплошных строк при переносе

```scss
 {
  overflow-wrap: normal;
  overflow-wrap: break-word; //мягкий разрыв предусматривается
  overflow-wrap: anywhere; //мягкий разрыв не предусматривается
}
```

# hyphens

указывает, как следует переносить слова через дефис, когда текст переносится на несколько строк

```scss
 {
  hyphens: none; //Слова не разрываются при переносе строки, даже если внутри слов указаны точки разрыва
  hyphens: manual; //Слова разрываются при переносе строки только там, где символы внутри слов указывают точки разрыва
  hyphens: auto; //Браузер может автоматически разбивать слова в соответствующих точках переноса, следуя любым правилам, которые он выбирает
  -moz-hyphens: auto;
  -ms-hyphens: auto;
  -webkit-hyphens: auto;
  //правильный разделитель слов (*)
  hyphens: auto;
}
```

# hyphenate-character

задает символ (или строку), используемый в конце строки перед переносом

```scss
.hyphenate-character {
  hyphenate-character: <string>;
  hyphenate-character: auto;
}
```

```html
<dl>
  <dt><code>hyphenate-character: "="</code></dt>
  <dd id="string" lang="en">Superc&shy;alifragilisticexpialidocious</dd>
  <dt><code>hyphenate-character is not set</code></dt>
  <dd lang="en">Superc&shy;alifragilisticexpialidocious</dd>
</dl>
```

```scss
dd {
  width: 90px;
  border: 1px solid black;
  hyphens: auto;
}

dd#string {
  -webkit-hyphenate-character: "=";
  hyphenate-character: "=";
}
```

# hyphenate-limit-chars (-ff -safari)

определяет минимальную длину слова, позволяющую переносить слова, а также минимальное количество символов до и после дефиса

```scss
.hyphenate-limit-chars {
  hyphenate-limit-chars: 10 4 4;
  hyphenate-limit-chars: 10 4;
  hyphenate-limit-chars: 10;

  /* Keyword values */
  hyphenate-limit-chars: auto auto auto;
  hyphenate-limit-chars: auto auto;
  hyphenate-limit-chars: auto;

  /* Mix of numeric and keyword values */
  hyphenate-limit-chars: 10 auto 4;
  hyphenate-limit-chars: 10 auto;
  hyphenate-limit-chars: auto 3;
}
```

# text-overflow

при переполнении текстом строки overflow: hidden; white-space: nowrap

```scss
 {
  // просто обрежет текст
  text-overflow: clip;
  // поставит троеточие (два значения для rtl)
  text-overflow: ellipsis ellipsis;
  text-overflow: ellipsis " [..]";
  text-overflow: ellipsis "[..] ";
}
```

Так же могут помочь символ `&shy` `<wbr>​`;

# text-align-last

Как выравнивается последняя строка в блоке или строка, идущая сразу перед принудительным разрывом строки.

```scss
.text-align-last {
  text-align-last: auto;
  text-align-last: start;
  text-align-last: end;
  text-align-last: left;
  text-align-last: right;
  text-align-last: center;
  text-align-last: justify;
}
```
