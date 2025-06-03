/**
 * @param {string} string
 *
 * @returns {string}
 */
const letterFrequency = (string) => {
  let result = "";
  const object = {};
  const stringLength = string.length;

  for (let i = 0; i < stringLength; i++) {
    const letter = string[i];

    object[letter] = object[letter] ? object[letter] + 1 : 1;
  }

  Object.entries(object)
    .sort(([letter1, frequency1], [letter2, frequency2]) =>
      frequency2 !== frequency1
        ? frequency2 - frequency1
        : letter2 < letter1
        ? 1
        : -1
    )
    .forEach(([letter, frequency]) => {
      result = `${result}${letter.repeat(frequency)}`;
    });

  return result;
};

console.log(letterFrequency("tree"));
