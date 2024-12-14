# ammonium string

Проверить строку не является ли она аммонием

```js
const isLetter = (char) => char.toLowerCase() === char.toUpperCase();

const isAmmoniumString = (string) => {
  let start = 0;
  let end = string.length - 1;

  while (start < end) {
    if (!isLetter(string[start])) {
      start += 1;
      continue;
    }

    if (!isLetter(string[end])) {
      end -= 1;
      continue;
    }

    if (string[start].toLowerCase() !== string[end].toLowerCase()) return false;

    start += 1;
    end -= 1;
  }

  return true;
};
```
