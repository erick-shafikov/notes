<!-- расширенные настройки начертания шрифтов openType --------------------------------------------------------------------------------------->

# расширенные настройки начертания шрифтов openType

Раньше все шрифты шли раздельно - «Roboto Regular», «Roboto Bold» и «Roboto Bold Italic». Что приводить к множественным запросам

керинг - растояние между определенными буквами
глиф - объединение букв ff fi и другие Лигатуры

## font-feature-settings

если шрифты имеют доп настройки (черточка в нуле итд), низкоуровневый доступ к шрифтам

```scss
.font-feature-settings {
  font-feature-settings: "smcp";
  font-feature-settings: "smcp" on;
  font-feature-settings: "swsh" 2;
  font-feature-settings: "smcp", "swsh" 2;
}
```

## font-language-override (-chrome, -safari, -ff)

переопределение очертания для других языков

```scss
.font-language-override {
  font-language-override: "ENG"; /* Use English glyphs */
  font-language-override: "TRK"; /* Use Turkish glyphs */
}
```

## font-optical-sizing

значения: none | auto - оптимизация глифов

## font-synthesis

Позволяет указать можно ли синтезировать жирно и прочее начертание

font-synthesis = font-synthesis-weight + font-synthesis-style + font-synthesis-small-caps + font-synthesis-position

```scss
.font-synthesis {
  font-synthesis: none;
  font-synthesis: weight; //Указывает, что отсутствующий жирный шрифт может быть синтезирован браузером при необходимости.
  font-synthesis: style; // Указывает, что курсивный шрифт может быть синтезирован браузером при необходимости.
  font-synthesis: position; //при необходимости подстрочный и надстрочный шрифт может быть синтезирован браузером при использовании
  font-synthesis: small-caps; //Указывает, что при необходимости браузер может синтезировать шрифт с малыми заглавными буквами.
  font-synthesis: style small-caps weight position; // property values can be in
}
```

далее ниже по два значения для каждого свойства

```scss
.font-synthesis {
  font-synthesis-style: auto; //можно, если нет и браузер попробует синтезировать
  font-synthesis-style: none; // нельзя
}
```

### font-synthesis-weight

может ли браузер синтезировать полужирное начертание

### font-synthesis-style

позволяет указать, может ли браузер синтезировать наклонный вариант

### font-synthesis-small-caps

позволяет указать, может ли браузер синтезировать шрифт с малыми капителями, если он отсутствует в семействе шрифтов

### font-synthesis-position (ff)

может ли браузер синтезировать подстрочные и надстрочные шрифты «position», если они отсутствуют в семействе шрифтов,

### palette-mix()

дял создания нового шрифта на основе двух других
