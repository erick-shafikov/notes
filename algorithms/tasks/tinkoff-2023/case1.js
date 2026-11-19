/* 


*/

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let res;

function isIncrease(arr) {
  let res = true;
  const length = arr.length;
  for (let i = 0; i < length; i++) {
    if (arr[i] > arr[i + 1]) {
      res = false;
    }
  }

  return res;
}

function isDecrease(arr) {
  let res = true;
  const length = arr.length;
  for (let i = 0; i < length; i++) {
    if (arr[i] < arr[i + 1]) {
      res = false;
    }
  }

  return res;
}

rl.on('line', (inputString) => {
  const arr = inputString.split(' ').map((item) => +item);
  res = isDecrease(arr) || isIncrease(arr) ? 'YES' : 'NO';
});

rl.on('close', () => {
  console.log(res);
  process.exit(0);
});
