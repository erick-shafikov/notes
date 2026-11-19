/**
 * @param {Array<number>} array
 */
const notDoubled = (array) => {
  const object = {};

  array.forEach((element) => {
    if (object[element]) {
      object[element] = object[element] + 1;
    } else {
      object[element] = 1;
    }
  });

  console.log(object);

  for (let key in object) {
    if (object[key] !== 2) return key;
  }
};

console.log(notDoubled([1, 2, 3, 2, 1]));
