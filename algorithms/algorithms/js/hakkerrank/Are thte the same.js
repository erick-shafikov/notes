a1 = [121, 144, 19, 161, 19, 144, 19, 11];
a2 = [
  11 * 11,
  121 * 121,
  144 * 144,
  19 * 19,
  161 * 161,
  19 * 19,
  144 * 144,
  19 * 19,
];

let fn = (a, b) => (+a > +b ? 1 : -1);

function comp(array1, array2) {
  for (let i = 0; i < array1.length; i++) {
    if (+array1.sort(fn)[i] * +array1.sort(fn)[i] != +array2.sort(fn)[i]) {
      return false;
    }
  }

  return true;
}
