# forwardRef

ref – на взывают рендер, позволяют обратится к предыдущему состоянию, единственное свойство current
Устанавливает на стадии коммита (до рендера), проблемы прикрепления ref и state можно решить с помощью flashSync()

Можно предотвращать ненужные рендеры

```tsx
function Search() {
  //исходный элемент, на который ссылаемся
  return <input type="search" />;
}

function App() {
  // ссылка на input  позволяет ссылаться на него
  const input = React.useRef<HTMLInputElement>(null); // создаем ссылку
  React.useEffect(() => {
    //фокусируемся при первом рендере
    if (input.current) {
      input.current.focus();
    }
  }, []);
  return <Search ref={input} />; // получаем проблему Property 'ref' does not exist on type 'IntrinsicAttributes'
}

// переопределенный
const Search = forwardRef<RefType, PropsType>((props, ref) => {
  const handleClick = () => {
    // костыль для ts
    if (modalRef != null && typeof modalRef !== "function") {
      modalRef?.current?.__someAction__();
    }
  };
  //решение – обернуть в forwardRef
  return <input ref={ref} type="search" />;
}); // получаем проблему 'ref‘ - unknown

// варианты типизации ---------------------------------------------------
const Component = React.forwardRef<RefType, PropsType>((props, ref) => {
  //пример решения
  return someComponent;
});

//или

const Search = React.forwardRef<HTMLInputElement>((props, ref) => {
  //решение – добавить generic
  return <input ref={ref} type="search" />;
});
```
