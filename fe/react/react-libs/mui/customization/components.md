# components

в createTheme можно изменить:

1. значение по умолчанию для всех компонентов
   - переписать стили темы
   - сделать состояние основанное на пропсах
   - сделать состояние основанное на стандартных атрибутах из апи компонентов
2. sx – проп
3. Создание новых вариантов для компонента
4. использование переменных темы

# кастомизация компонентов

1. Одноразовая кастомизация с помощью sx пропсы. Переопределение стилей вложенных элементов в стандартные компоненты строится на том, что каждый класс строится по принципу [hash]-Mui[Component name]-[name of the slot].
2. classNames – атрибут. Для каждого псевдо класса у MUI есть свой класс для более высокой специфики active - .Mui-active, checked - .Mui-checked, completed - .Mui-completed, disabled - .Mui-disabled, error - .Mui-error, expanded - .Mui-expanded, focus visible - .Mui-focusVisible, focused - .Mui-focused, readOnly - .Mui-readOnly, required - .Mui-required, selected - .Mui-selected
3. Переисользуемые компоненты
   - без пропсов
   - с пропсами, которые позволяют динамически изменять компонент
   - с переопределяемым объектом стилей, в котором переопределяются значение переменных CSS
4. Переписывание глобальных стилей с помощью темы
5. Переопределение с помощью компонента `<GlobalStyles styles={…} />`. BP для глобальных стилей является определение компонента в переменную

```tsx
const inputGlobalStyles = <GlobalStyles styles={} />;
function Input(props) {
  return (
    <React.Fragment>
      {inputGlobalStyles}
      <input {...props} />     
    </React.Fragment>
  );
}
```

# api design approach

Композиция:

- children
- пропы-children-ы
- spread – пропсы передаются от родителя к потомку до root, нежелательно использовать classNames
- классы всегда применяются к root элементы
- стили по умолчанию сгруппированы в один класс
- все стили не применяемые к root имеют префикс
- булевы значения идут без префикса

CSS – классы:

- класс применённый к root элементу называется root
- классы по умолчанию формируют один класс
- Остальные классы имеют префикс
- булевы не имеют префикса
- enum-свойства имеют префикс

Вложенные компоненты имеют

- свой пропы (id для Input)
- xxxProp
- xxxComponent
- xxxRef

Наименование
Если два значения – boolean, если больше, то Enum

# совместимость

CSS
через импорт .css фалов, в которых указаны классы компонентов (для одноразовых компонентов)
Глобальный CSS
Нужно пользоваться встроенным классами .MuiSlider-root, .MuiSlider-root:hover и другими
CSS-модули (аналогично CSS)
Styled
Тема
Emotion - через css-проп

# композиция

При оборачивании компонентов

```jsx
const WrappedIcon = (props) => <Icon {...props} />;
WrappedIcon.muiName = Icon.muiName;
```

component-проп
Обеспечивает полиморфные компоненты, если в качестве компонента указать inline кастомный компонент, то это повлечет за собой потерю состояния, деструктуризация компонента при каждом рендере. Выход обернуть в useMemo

Если передаем react-комопнент, а не обычный html – элемент, то нужно его обернуть в memo

```tsx
import { Link, LinkProps } from "react-router-dom";

function ListItemLink(props) {
  const { icon, primary, to } = props;
  const CustomLink = React.useMemo(
    () =>
      React.forwardRef<HTMLAnchorElement, Omit<RouterLinkProps, "to">>(
        function Link(linkProps, ref) {
          return <Link ref={ref} to={to} {...linkProps} />;
        }
      ),
    [to]
  );

  return (
    <li>
      <ListItem button component={CustomLink}>
        <ListItemIcon>{icon}</ListItemIcon>
        <ListItemText primary={primary} />
      </ListItem>
    </li>
  );
}
```

## роутинг

Для react-router-dom и next-link есть готовые решения
