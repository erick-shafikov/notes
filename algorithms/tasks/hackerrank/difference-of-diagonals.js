let arr = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];

function diagonalDifference(arr) {
  let mainSum = 0;
  let secondSum = 0;

  for (let i = 0; i < arr.length; i++) {
    mainSum += arr[i][i];
    secondSum += arr[i][arr[i].length - 1 - i];
  }

  return Math.abs(mainSum - secondSum);
}

alert(diagonalDifference(arr));
