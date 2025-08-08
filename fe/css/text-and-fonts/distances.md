# word-spacing

```scss
.word-spacing {
  word-spacing: 3px;
  word-spacing: 0.3em;

  /* <percentage> значения  */
  word-spacing: 50%;
  word-spacing: 200%;
}
```

# letter-spacing

расстояние между буквами

```scss
 {
  letter-spacing: "px", "%";
}
```

# tab-size

размер символа табуляции

# text-indent

определяет размер отступа (пустого места) перед строкой в текстовом блоке.

# white-space

Свойство white-space управляет тем, как обрабатываются пробельные символы внутри элемента.

```scss
 {
  white-space: normal; //Последовательности пробелов объединяются в один пробел.
  white-space: nowrap; //не переносит строки (оборачивание текста) внутри текста. Удалит переносы строк
  white-space: pre; //Последовательности пробелов сохраняются так, как они указаны в источнике.
  white-space: pre-wrap; //как и в pre + <br/>
  white-space: pre-line; //только <br />
  white-space: break-spaces;
}
```

- !!! white-space: pre-wrap позволяет в react, в пропсы передавать текст с пробельными символами (перенос строки итд)

```scss
.container {
  white-space: pre-wrap;
}
```

```jsx
const Comp = ({ text }) => <div className="container">{text}</div>;

<Comp text={"текст\nЕще текст"} />;
```

# white-space-collapse

управляет тем, как сворачивается пустое пространство внутри элемента

```scss
.white-space-collapse {
  white-space-collapse: collapse;
  white-space-collapse: preserve;
  white-space-collapse: preserve-breaks;
  white-space-collapse: preserve-spaces;
  white-space-collapse: break-spaces;
}
```

# font-kerning

расстояние между буквами

```scss
.font-kerning {
  font-kerning: auto;
  font-kerning: normal;
  font-kerning: none;
}
```
