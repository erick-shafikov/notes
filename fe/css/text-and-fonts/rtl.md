<!-- Направление письма ---------------------------------------------------------------------------------------------------------------------->

# направление письма

Блочная модель так же предусматривает направление текста

## writing-mode

устанавливает горизонтальное или вертикальное положение текста также как и направление блока

```scss
.writing-mode {
  writing-mode: horizontal-tb; // поток - сверху вниз, предложения - слева направо
  writing-mode: vertical-rl; // поток - справа налево, предложения - вертикально
  writing-mode: vertical-lr; // поток - слева направо, предложения - вертикально
}
```

## direction

принимает два значения ltr и rtl

## text-orientation

позволяет распределить символы в вертикальном и горизонтальном направлениях

```scss
.text-orientation {
  text-orientation: mixed;
  text-orientation: upright; //сверху вниз
  text-orientation: sideways-right;
  text-orientation: sideways;
  text-orientation: use-glyph-orientation;
}
```

## text-combine-upright

учет чисел при написании в иероглифах all - все числа будут упакованы в размер одного символа

## line-break

перенос китайского и японского
