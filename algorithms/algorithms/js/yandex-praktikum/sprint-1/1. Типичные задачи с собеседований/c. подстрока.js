const substring = (string) => {
  let maxLength = -Infinity;
  const stringLength = string.length;

  const charsArray = [];
  let winner = "";

  let start = 0;
  let end = start++;

  while (end < stringLength) {
    const letter = string[start];
    const index = charsArray.indexOf(letter);

    if (index !== -1) {
      charsArray.splice(0, index);

      start = start + index + 1;
      end = start + 1;
    } else {
      charsArray.push(letter);

      if (charsArray.length > maxLength) {
        maxLength = charsArray.length;
        winner = charsArray.join("");
      }

      start++;
    }
  }

  return { maxLength, winner };
};

console.log(substring("aaaaaaaaaaabcdefeeee"));
console.log(substring("abcdabcbb"));
