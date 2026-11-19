/*
Вход 769082435 210437958 673982045 375809214 380564127
Выход 1640793344 2199437821 1735893734
*/

function compare(a, b) {
  if (a > b) return 1;
  if (a === b) return 0;
  if (a < b) return -1;
}

arr = [769082435, 210437958, 673982045, 375809214, 380564127];

function minMaxSum(array) {
  let maxSum = 0;
  let minSum = 0;

  array.sort(compare);

  for (let i = 0; i < array.length - 1; i++) {
    minSum += array[i];
    maxSum += array[array.length - i - 1];
  }

  console.log(`${minSum} ${maxSum}`);
}
