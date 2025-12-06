# fieldset

Форма может содержать один или несколько элементов fieldset

```html
<body>
  <form id="form">
    <!-- открыть блок -->
   <fieldset name="userFields">
     <!-- подпись блока -->
      <legend>info</legend>
      <input name="login" type="text">
   </fieldset>
   </form>
  </body>

<script>
  alert(form.elements.login); //<input name="login">
  let fieldset = form.elements.userFields; //fieldset"ы доступны так же как элемент в elements
  alert(fieldset); //HTMLFieldsetElement
  alert(fieldset.elements.login == form.elements.login); //true
<script>

```
