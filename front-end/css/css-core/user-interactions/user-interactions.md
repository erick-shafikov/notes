# resize (-s)

позволяет сделать элемент растягиваемым

```scss
.resize {
  resize: none; //отключает растягивание
  resize: both; //тянуть можно во все стороны
  resize: horizontal;
  resize: vertical;
  resize: block; // в зависимости от writing-mode и direction
  resize: inline; // в зависимости от writing-mode и direction
}
```

# user-select

метод выделения текста курсором

[user-select](./text.md#user-select)

# overlay (-sf, -ff)

определяет, отображается ли элемент, появляющийся в верхнем слое (например, показанный поповер или модальный элемент), на самом деле в верхнем слое в элементе dialog

значения
overlay: auto;
overlay: none;

## ::highlight()

для выделения текста

## ::selection

для выделенной части
