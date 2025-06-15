# render prop

- передача в качестве props функции, которая будет принимать какие-то данные от дочернего компонента и отрисовывать их так, как будет указано в родительском

```jsx
const ParentComponent = () => {
  //получаем в качестве пропса функцию на отрисовку с аргументами
  return <ChildComponent render={(text) => <h1>{text}</h1>} />;
};

const ChildComponent = ({ render }) => {
  const text = "Hello World";
  //аргументами, которые можно передать из дочернего
  return <div>{render(text)}</div>;
};
```

counter

```js
const ClickCounter = ({ render }) => {
  const [count, setCount] = useState(0);

  const increment = () => setCount(count + 1);

  return <div>{render({ count, increment })}</div>;
};

<ClickCounter
  render={({ count, increment }) => (
    <div>
      <h2>Кастомный счётчик</h2>
      <p>Количество кликов: {count}</p>
      <button onClick={increment}>Прибавить 1</button>
    </div>
  )}
/>;
```

Форма, в которой если не передан render отрисует форму по умолчанию

```js
const Form = ({ initialValues, render }) => {
  const [values, setValues] = useState(initialValues);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues((previousValues) => ({ ...previousValues, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log("Отправленные значения", values);
  };

  //если предан, то отрисуется форма, которую передали
  if (render) {
    return render({
      values,
      handleChange,
      handleSubmit,
    });
  }

  //иначе по умолчанию
  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(initialValues).map((key) => (
        <div key={key}>
          <label>
            <div>{key[0].toUpperCase() + key.slice(1)}:</div>
            <input
              type="text"
              name={key}
              value={values[key]}
              onChange={handleChange}
            />
          </label>
        </div>
      ))}
      <button type="submit">Отправить</button>
    </form>
  );
};
```
