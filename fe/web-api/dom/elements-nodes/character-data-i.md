# свойства экземпляра

## text.data

значение текстового поля, при "" === null

```html
 
<body>
      Привет    
  <!-- Комментарий -->
     
  <script>
    let text = document.body.firstChild;
    alert(text.data); //Привет
    let comment = text.nextSibling;
    alert(comment.data); //Комментарий
  </script>
   
</body>
```
