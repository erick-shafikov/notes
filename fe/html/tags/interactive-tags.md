<!-- a ----------------------------------------------------------------------------------------------------------------------->

# a (inline)

!!!Ссылки являются встроенным элементом, внутри тега a нельзя располагать блочные элементы и наоборот вкладывать ссылку в блочный контейнер

аттрибуты:

- href единственный обязательный атрибут. mail.to Создание
  ссылки на электронный адрес электронной почты с атрибутом mailto:адрес эл почты,
  при нажатии запускается почтовая программа, можно добавить параметр через ?
  subject = тема сообщения

```html
<a href="mailto:name@mail.ru?subject=Тема письма">Задавайте вопросы</a>

<a
  href="mailto:nowhere@mozilla.org?cc=name2@rapidtables.com&bcc=name3@rapidtables.com&amp;subject=The%20subject%20of%20the%20email &amp;body=The%20body%20of%20the%20email"
>
  Отправить письмо с полями cc, bcc, subject и body
</a>
```

Также параметрами строки могут выступать «subject», «cc» и «body»

- download - если есть значение у этого атрибута, то файл будет скачен с таким именем
- hreflang - язык документа по ссылке
- ping - уведомляет указанные в нём URL, что пользователь перешёл по ссылке
- referrerpolicy какую информацию передавать по ссылке
- - no-referrer - без заголовка Referer
- - no-referrer-when-downgrade - не отправляет заголовок Referer ресурсу без TLS HTTPS
- - origin - отправит информацию о странице адрес итд
- - origin-when-cross-origin - путь отправит только внутри ресурса
- - unsafe-url - отправляет только ресурс и адрес
- rel - устанавливает отношения между ссылками
- target - по умолчанию открывается в текущем окне или фрейме, но можно изменить с помощью target

```html
<a target="Имя окна">…</a>

<!-- в качестве значения используется имя окна или фрейма. -->
<!-- Зарезервированные имена:  -->
<!-- _blank – загружает страницу в новое окно браузера-->
<a target="_blank"></a>
<!-- загружает страницу в текущее окно  -->
<a target="_self"></a>
<!--_parent загружает страницу во фрейм-родитель-->
<a target="_parent"></a>
<!--_top отменяет все фреймы-->
<a target="_top"></a>
<!--для работы с фреймами-->
<a target="_unfencedTop"></a>
```

- title - дополнительная информация о ссылке (будет отображаться)

```html
<a href="link" title="дополнительная информация при наведении"></a>
```

- type - определяет MIME тип

- устаревшие: charset, coords, name, rev, shape
- нестандартные: datafld, datasrc, methods, urn

## Якоря

Якорь – закладка с уникальным именем на определенном месте страницы, для создания перехода к ней

```html
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html" charset="utf-8" />
    <title>Быстрый доступ внутри документа</title>
  </head>
  <body>
    <p><a name="top"></a></p>
    <p>…</p>
    <p><a href="#top">Наверх </a></p>
    <p></p>
  </body>
</html>
```

в общем виде

```html
<div id="id">Объект навигации</div>

<a href="document-name.html#id">наведет на div выше</a>
```

## BP. Создание кликабельной картинки

```html
<a href="https://developer.mozilla.org/ru/" target="_blank">
  <img src="mdn_logo.png" alt="MDN logo" />
</a>
```

## BP. Создание ссылки с номером телефона

```html
<a href="tel:+491570156">+49 157 0156</a>
```

## BP. сохранение рисунка canvas

```js
var link = document.createElement("a");
link.innerHTML = "download image";

link.addEventListener(
  "click",
  function (ev) {
    link.href = canvas.toDataURL();
    link.download = "myPainting.png";
  },
  false
);

document.body.appendChild(link);
```

<!-- details ----------------------------------------------------------------------------------------------------------------->

# details и summary (block, HTML5)

Раскрывающееся меню вниз

[summary - тег будет использован как заголовок](#summary-html5)

```html
<details>
  <summary>Details</summary>
  Something small enough to escape casual notice.
</details>
```

Атрибуты:

- open - изначальное состояние

стилизовать маркер можно с помощью ::-webkit-details-marker
summary {display: block;} для того что бы скрыть треугольник по умолчанию и добавить свой

Добавление собственного элемента для summary

```html
<details>
  <summary>Some details</summary>
  <p>More info about the details.</p>
</details>
```

```scss
// выключить маркер
summary {
  display: block;
}

summary::-webkit-details-marker {
  display: none;
}

summary::before {
  content: "\25B6";
  padding-right: 0.5em;
}

details[open] > summary::before {
  content: "\25BC";
}
```

## вложенный горизонтальный список

```html
<div>
  <details name="starWars" open>
    <summary>Prequels</summary>
    <ul>
      <li>Episode I: The Phantom Menace</li>
      <li>Episode II: Attack of the Clones</li>
      <li>Episode III: Revenge of the Sith</li>
    </ul>
  </details>

  <details name="starWars">
    <summary>Originals</summary>
    <ul>
      <li>Episode IV: A New Hope</li>
      <li>Episode V: The Empire Strikes Back</li>
      <li>Episode VI: Return of the Jedi</li>
    </ul>
  </details>

  <details name="starWars">
    <summary>Sequels</summary>
    <ul>
      <li>Episode VII: The Force Awakens</li>
      <li>Episode VIII: The Last Jedi</li>
      <li>Episode IX: The Rise of Skywalker</li>
    </ul>
  </details>
</div>
```

```scss
div {
  gap: 1ch;
  display: flex;
  position: relative;

  details {
    min-height: 106px; /* Prevents content shift */

    &[open] summary,
    &[open]::details-content {
      background: #eee;
    }

    &[open]::details-content {
      //каждый список прилипнет к левому краю
      left: 0;
      position: absolute;
    }
  }
}
```

```scss
//все тоже самое, но через якоря
div {
  display: inline-grid;
  anchor-name: --wrapper;

  details[open] {
    summary,
    &::details-content {
      background: #eee;
    }

    &::details-content {
      position: absolute;
      position-anchor: --wrapper;
      top: anchor(top);
      left: anchor(right);
    }
  }
}
```

## обработка с помощью js

```js
// Перебираем раскрывающиеся элементы
document.querySelectorAll("details").forEach((details) => {
  // Обрабатываем переключение видимости контента
  details.addEventListener("toggle", () => {
    if (details.open) {
      // Элемент открыт
    } else {
      // Элемент закрыт
    }
  });
});
```

<!-- dialog ----------------------------------------------------------------------------------------------------------------->

# dialog (block, HTML5)

Элемент диалогового окна. Пример с окном выбора email. нельзя присваивать tabIndex
Атрибуты:

- open

::backdrop - позволяет стилизовать подложку

```scss
// использование
::backdrop {
  background: hsl(0 0 0 / 90%);
  backdrop-filter: blur(
    3px
  ); /* Интересное свойство, предназначенное только для фонов */
}
```

Пример формы

```html
<!-- Простой попап диалог с формой -->
<dialog id="favDialog">
  <form method="dialog">
    <section>
      <p>
        <label for="favAnimal">Favorite animal:</label>
        <select id="favAnimal">
          <option></option>
          <option>Brine shrimp</option>
          <option>Red panda</option>
          <option>Spider monkey</option>
        </select>
      </p>
    </section>
    <menu>
      <button id="cancel" type="reset">Cancel</button>
      <button type="submit">Confirm</button>
    </menu>
  </form>
</dialog>

<menu>
  <button id="updateDetails">Update details</button>
</menu>

<script>
  (function () {
    var updateButton = document.getElementById("updateDetails");
    var cancelButton = document.getElementById("cancel");
    var favDialog = document.getElementById("favDialog");

    // Update button opens a modal dialog
    updateButton.addEventListener("click", function () {
      favDialog.showModal();
    });

    // Form cancel button closes the dialog box
    cancelButton.addEventListener("click", function () {
      favDialog.close();
    });
  })();
</script>
```

скрипт для функциональности каждого модального окна. data атрибут нужен для идентификации каждого

```js
// Перебираем все элементы с атрибутом data-dialog
document.querySelectorAll("[data-dialog]").forEach((button) => {
  // Обрабатываем взаимодействие (клик)
  button.addEventListener("click", () => {
    // Выбираем соответствующее диалоговое окно
    const dialog = document.querySelector(`#${button.dataset.dialog}`);
    // Открываем его
    dialog.showModal();
    // Закрываем
    dialog
      .querySelector(".closeDialog")
      .addEventListener("click", () => dialog.close());
  });
});
```

предотвращение появления полосы прокрутки

```scss
body:has(dialog:modal) {
  overflow: hidden;
}
```

# popover api

всплывающие окна

```html
<button popovertarget="tooltipA">Show tooltipA</button>

<div id="tooltipA" popover>
  <button popovertarget="tooltipA">Hide tooltipA</button>
</div>
```

использование с dialog, по молчанию располагаются посередине

```html
<main>
  <button popovertarget="tooltipA">Show tooltipA</button>
</main>

<dialog id="tooltipA" popover>
  <button popovertarget="tooltipA">Hide tooltipA</button>
</dialog>
```

для того чтобы сместить с середины

```scss
main [popovertarget] {
  anchor-name: --trigger;
}

[popover] {
  margin: 0;
  position-anchor: --trigger;
  top: calc(anchor(bottom) + 10px);
  justify-self: anchor-center;
}
```

Пример с js

```html
<main>
  <button id="anchorLink" popovertarget="tooltipLink">Open tooltipLink</button>
  <button id="anchorNoLink" popovertarget="tooltipNoLink">
    Open tooltipNoLink
  </button>
</main>

<dialog anchor="anchorLink" id="tooltipLink" popover>
  Has <a href="#">a link</a>, so we can’t hide it on mouseout
  <button popovertarget="tooltipLink">Hide tooltipLink manually</button>
</dialog>

<dialog anchor="anchorNoLink" id="tooltipNoLink" popover>
  Doesn’t have a link, so it’s fine to hide it on mouseout automatically
  <button popovertarget="tooltipNoLink">Hide tooltipNoLink</button>
</dialog>
```

```scss
[popover] {
  margin: 0;
  position: fixed;
  top: calc(anchor(bottom) + 10px);
  justify-self: anchor-center;

  /* No link, no button needed */
  &:not(:has(a)) [popovertarget] {
    display: none;
  }
}
```

```js
/* Перебираем все триггеры поповеров */
document.querySelectorAll("main [popovertarget]").forEach((popovertarget) => {
  /* Выбираем соответствующий поповер */
  const popover = document.querySelector(
    `#${popovertarget.getAttribute("popovertarget")}`
  );

  /* Отображаем поповер при наведении курсора на триггер */
  popovertarget.addEventListener("mouseover", () => {
    popover.showPopover();
  });

  /* Скрываем поповер при снятии курсора с триггера, если он не содержит ссылку */
  if (popover.matches(":not(:has(a))")) {
    popovertarget.addEventListener("mouseout", () => {
      popover.hidePopover();
    });
  }
});
```

реализация с фоном

```html
<!-- Re-showing ‘A’ rolls the onboarding back to that step -->
<button popovertarget="onboardingTipA" popovertargetaction="show">
  Restart onboarding
</button>
<!-- Hiding ‘A’ also hides subsequent tips as long as the popover attribute equates to auto -->
<button popovertarget="onboardingTipA" popovertargetaction="hide">
  Cancel onboarding
</button>

<ul>
  <li id="toolA">Tool A</li>
  <li id="toolB">Tool B</li>
  <li id="toolC">Another tool, “C”</li>
  <li id="toolD">Another tool — let’s call this one “D”</li>
</ul>

<!-- onboardingTipA’s button triggers onboardingTipB -->
<dialog anchor="toolA" id="onboardingTipA" popover>
  onboardingTipA
  <button popovertarget="onboardingTipB" popovertargetaction="show">
    Next tip
  </button>
</dialog>

<!-- onboardingTipB’s button triggers onboardingTipC -->
<dialog anchor="toolB" id="onboardingTipB" popover>
  onboardingTipB
  <button popovertarget="onboardingTipC" popovertargetaction="show">
    Next tip
  </button>
</dialog>
```

```scss
::backdrop {
  animation: 2s fadeInOut;
}

[popover] {
  margin: 0;
  position: fixed;
  align-self: anchor-center;
  left: calc(anchor(right) + 10px);
}

/* Not important */
ul {
  gap: 30px;
  padding: 0;
  display: grid;

  li {
    width: max-content;
  }
}

@keyframes fadeInOut {
  0% {
    background: hsl(0 0 0 / 0%);
  }
  25% {
    background: hsl(0 0 0 / 50%);
  }
  50% {
    background: hsl(0 0 0 / 50%);
  }
  75% {
    background: hsl(0 0 0 / 50%);
  }
  100% {
    background: hsl(0 0 0 / 0%);
  }
}
```

```js
setTimeout(() => {
  document.querySelector("#onboardingTipA").showPopover();
}, 2000);
```

<!-- summary ------------------------------------------------------------------------------------------------------------------->

# summary (HTML5)

Видимы заголовок для [details](#details-и-summary-block-html5)

display: list-item
