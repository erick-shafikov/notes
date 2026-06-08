# DocumentFragment

является оберткой для передачи списков узлов, мы можем добавить к нему другие узлы

getListContent генерирует фрагмент с элементами <li>, которые позже вставляются в <ul>

```html
<ul id="ul"></ul>

<script>
  function getListContent() {
    let fragment = new DocumentFragment(); //создаем вставляемый фрагмент

    for (let i = 1; i <= 3; i++) {
      let li = document.createElement("li"); //создаем li элемент
      li.append(i); //создать номера в списке
      fragment.append(li); //вставляем во фрагмент элемент li c номером
    }

    return fragment; //возвращаем результат функции
  }

  ul.append(getListContent()); //он исчезнет (не добавится) вместо него вставится список с тремя li
</script>
```
