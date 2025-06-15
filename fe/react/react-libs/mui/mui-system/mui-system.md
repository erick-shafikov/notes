# MUI - system

Основной метод кастомизации в MUI
Основывается на sx пропсе, позволяет избежать вложенности контекста и лишних компонентов
Минус заключается в том, что компоненты с sx самые затратные по производительности

SX пропс предоставляет все CSS свойства, а также кастомные. Поддерживает css – свойства, такие как hover, media-queries, вложенные селекторы
Box – основной элемент для кастомизации

Утилиты затрагивают такие аспекты как border, display, flex-box, grid, palette

Основные компоненты (у всех есть element, sx, children пропсы):

- Box (базовый div-компонент)
- Container (позволяет центровать горизонтально контент, fluid – проп позволяет растянуть на всю ширину, fixed – будет синхронизироваться с breakpoint), props { classes, disableGutters: boolean (уберет padding-и слева и справа), fixed, maxWidth: BreakPoints}
- Grid (основной элемент, который позволяет контролировать сетку), props {columns: number (количество колонок), columnSpacing, container, direction, disableEqualOverflow, lg, lgOffset … xs xsOffset (сколько колонок занимает / сколько колонок пропустить ), wrap, }
- Stack (вертикальная сетка) props {direction, divider: node, spacing, useFlexGap}

# breakpoints

вариант объектом

```js
sx={{
  width: {
    xs: 100, // theme.breakpoints.up('xs')
    sm: 200, // theme.breakpoints.up('sm')
    md: 300, // theme.breakpoints.up('md')
    lg: 400, // theme.breakpoints.up('lg')
    xl: 500, // theme.breakpoints.up('xl')
  },
}}

```

2. Вариант - массивом

```js
<Box sx={{ width: [100, 200, 300] }}>This box has a responsive width.</Box>
```

# BP

## imports

Импорты

```js
// 🐌 Named
import { Delete } from "@mui/icons-material";
// 🚀 Default
import Delete from "@mui/icons-material/Delete";
```

## props

sx проп менее производительный вариант, чем указание стилевых проп (системные UI-пропы)
Все значения темы доступны через styled, как параметр theme
