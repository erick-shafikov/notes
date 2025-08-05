# вложенность

```scss
.navigation {
  // .navigation li {…}
  list-style: none;

  li {
    display: inline-block;
    margin: 30px;
  }
}

.navigation {
  //с псевдо классами
  list-style: none;

  li {
    display: inline-block;
    margin-left: 30px;

    &:first-child {
      margin: 0;
    }
  }
}
```
