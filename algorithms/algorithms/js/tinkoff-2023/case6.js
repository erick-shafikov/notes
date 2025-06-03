const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let inputArr = [];
let res;

rl.on('line', (inputItem) => {
  inputArr.push(inputItem);

  const [qnt, maxSum] = inputArr[0].split(' ').map((item) => +item);
  let median = 0;

  for (let i = 1; i < inputArr.length - 1; i++) {
    let tempArr = inputArr[i].split(' ').map((item) => +item);
    median += (tempArr[0] + tempArr[1]) / 2;
  }

  res = Math.floor(median / qnt + 2);
});

rl.on('close', () => {
  console.log(res);
  process.exit(0);
});
