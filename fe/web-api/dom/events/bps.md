## Меню

Все кнопки являются ссылками, но при использовании ПКМ и открыть в новом окне это работать не будет

```html
<ul id="menu" class="menu">
  <li><a href="/html">HTML</a></li>
  <li><a href="/javascript">JavaScript</a></li>
  <li><a href="/css">CSS</a></li>
</ul>

<script>
  //необходимо обработать клики в JS, а стандартное действие переход – отменить
  menu.onclick = function (event) {
    if (event.target.nodeName != "a") return;

    let href = event.target.getAttribute("href"); //получить ссылку
    alert(href); //что-то сделать с ней

    return false; //отменяем стандартное действие – переход по ссылки
  };
</script>

<!-- фокус работает -->
<input value="Фокус работает" onfocus="this.value=''" />
<!-- фокус не работает -->
<input onmousedown="return false" onfocus="this.value" value="Кликни меня" />
```
