<!-- Позиционирование -------------------------------------------------------------------------------------------------------------------->

# Позиционирование

Используете для расположения декоративных элементов

- static - по умолчанию, где находятся, как будто без CSS
- absolute - абсолютное – позиционирование относительно родителя, при передвижении двигается весь документ, ориентация относительно родителя для всех элементов, которые находятся внутри него, при этом другие элементы отображаются на веб-странице словно абсолютно позиционированного элемента и нет (выкидывает из потока).
  Положение элемента задается свойствами left, top, right и bottom, также на положение влияет значение свойства position родительского элемента. Так, если у родителя значение position установлено как static или родителя нет, то отсчет координат ведется от края окна браузера. Если у родителя значение position задано как fixed, relative или absolute, то отсчет координат ведется от края родительского элемента.
  fixed
- relative относительно родителя, само по себе ничего не делает, режим позиционирования top, right, bottom, left относительно родителя, при движении другие элементы не будут сдвинуты, не выкидывает его из потока, но двигает поверх других элементов в потоке
- fixed - фиксированное, остается на том же месте, при прокрутке, позиционируется от родителя
- sticky - закрепленное, липкое

- z-index – управление порядком слоев при перекрытие

Вкратце:

- absolute – выкидывает из потока и позиционирует относительно родителя
- relative – не выкидывает из потока и позиционирует относительно родителя
- fixed – вообще не двигается
- sticky – двигается во фрейме

Значения для top? left, right, bottom - любые от пикселей до rem

## Выравнивание элементов

выравнивание происходит по двум осям inline - main, block - cross, выравнивание по главной: justify-items,justify-self, justify-content. По поперечной: align-items, align-self, align-content. Выделяют контейнер выравнивания, элемент выравнивания, запасное выравнивание
Типы выравнивания: Positional alignment (выравнивание положения - start, end, center, left...), Baseline alignment (исходное выравнивание baseline, first baseline, last baseline), Distributed alignment (распределённое выравнивание stretch, space-between, space-around, space-evenly)

- [основой для выравнивания могут служить сетки](#сетки-flex)
- [свойство justify-items которое позволяет выравнивать элементы в обычном блоке](./css-props.md/#justify-item)
- [vertical-align вертикальное выравнивание](./css-props.md/#vertical-align)

Применимо для выравнивание текста и изображения в одном контейнере или в таблице

```scss
img.top {
  vertical-align: text-top;
}
img.bottom {
  vertical-align: text-bottom;
}
img.middle {
  vertical-align: middle;
}
```

```html
<div>
  Изображение <img src="frame_image.svg" alt="link" width="32" height="32" /> с
  выравниванием по умолчанию.
</div>
<div>
  Изображение
  <img class="top" src="frame_image.svg" alt="link" width="32" height="32" /> с
  выравниванием по верхнему краю.
</div>
<div>
  Изображение
  <img class="bottom" src="frame_image.svg" alt="link" width="32" height="32" />
  с выравниванием по нижнему краю.
</div>
<div>
  Изображение
  <img class="middle" src="frame_image.svg" alt="link" width="32" height="32" />
  с выравниванием по центру.
</div>
```

## BP. Центрирование с помощью position

```css
.centered {
  /* абсолютно позиционируем */
  position: absolute;
  /* сверху, относительно родителя */
  top: 40%;
  /* снизу, относительно родителя */
  left: 50%;
  /* двигать относительно самого элемента} */
  transform: translate(-50%, -50%);
}
```
