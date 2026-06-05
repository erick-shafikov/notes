# Элементы формы

select имеет 3 свойства:

- select.options – коллекция из под-элементов option
- select.value – значение выбранного в данный момент option
- select.selectedIndex – номер выбранного option

они дают три способа установить значение в select:

- Найти соответствующий элемент option и установить в option.selected Значение true
- установить в select.value – значение выбранного в данный момент option
- установить в select.selectedIndex номер нужного option

```html
<!-- три варианта выбрать банан, как стартовый пункт в всплывающем меню -->
<select id="select">
  <option value="apple">Яблоко</option>

  <option value="pear">Груша</option>
  <option value="banana">Банан</option>
</select>

<script>
  select.options[2].selected = true; //как правило такой вариант не выбирают
  select.selectedIndex = 2;
  select.value = "banana";
</script>
```

select позволяет нам выбрать несколько вариантов одновременно, если у него стоит атрибут multiple

```html
<select id="" select="" multiple>
  <option value="" blues="" selected>Блюз</option>
  <option value="" rock="" selected>Рок</option>
  <option value="" classic="">Классика</option>
</select>

<script>
  let selected = Array.from(select.option)
    .filter((option) => option.selected) //добавляет если true
    .map((option) => option.value); //меняет элемент массивов на value

  alert(selected);
</script>
```

# new Option

Синтаксис создания элемента option

```js
option = new Option(text.value, defaultSelected, selected);
```

- text – текст внутри option
- value – значение
- defaultSelected – если true то ставится HTML атрибут selected
- selected – если true то элемент option будет выбранным

```js
let option = new Option("Текст", "value"); //создаст <option value="value">Текст</option>
let option = new Option("Текст", "value", true, true); //такой же элемент только выбранный в списке
```
