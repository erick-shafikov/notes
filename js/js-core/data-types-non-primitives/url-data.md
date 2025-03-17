```js
// decodeURI() - декодирование
const uri = "https://mozilla.org/?x=шеллы";
// encodeURI
const encoded = encodeURI(uri); // Expected output: "https://mozilla.org/?x=%D1%88%D0%B5%D0%BB%D0%BB%D1%8B"
decodeURI(encoded); //Expected output: "https://mozilla.org/?x=шеллы"
encodeURIComponent(); //Позволяет закодировать все символы
encodeURI(); //позволяет кодировать основные символы
```
