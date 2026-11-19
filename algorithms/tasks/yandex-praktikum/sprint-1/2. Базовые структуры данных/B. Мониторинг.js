/**

 * @param {number[][]} array 
 */
const swapArray = (array) => {
  const resultArray = [];
  for (let i = 0; i < array.length; i++) {
    const nestedArrayLength = array[i].length;
    for (let j = 0; j < nestedArrayLength; j++) {
      if (!resultArray[j]) {
        resultArray[j] = [];
      }

      resultArray[j][i] = array[i][j];
    }
  }

  return resultArray;
};

console.log(
  swapArray([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
  ])
);
