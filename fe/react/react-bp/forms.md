# BP. forms

```js
// Пример формы в виде функционального компонента

function App() {
  const [name, setName] = useState("");
  const [headingText, setHeading] = useState("");
  function handleChange(e) {
    setName(e.target.value);
  }
  const onClickHandler = (e) => {
    setHeading(name);
    e.preventDefault();
  };
  return (
    <div className="container">
      <h1>Hello {headingText}</h1> 
      <form onSubmit={onClickHandler}>
         
        <input
          onChange={handleChange}
          type="text"
          placeholder="What's your name?"
          value={name}
        />
        <button type="submit">Submit</button> 
      </form>
    </div>
  );
}
```
