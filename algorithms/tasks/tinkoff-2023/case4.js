const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let res;
let inputArr = [];

rl.on('line', (inputString) => {
  inputArr.push(inputString);

  if (inputArr.length === 2) {
    const [length, array] = inputArr;
    const numbers = array.split(' ');
    const obj = {};

    for (let i = 0; i < length; i++) {
      if (numbers[i] in obj) {
        obj[numbers[i]] += 1;
      } else {
        obj[numbers[i]] = 1;
      }
    }

    let qnt = Object.values(obj);
    let lengthQnt = qnt.length;
    let max = 0;
    let temp = 0;

    for (let i = 0; i < lengthQnt; i++) {
      let item = qnt[i] - 1;
      for (let j = 0; j < lengthQnt; j++) {
        if (item === qnt[j]) {
          temp += qnt[j];
        }
      }
      max = temp > max ? temp + item + 1 : max;
      temp = 0;
    }

    res = max > lengthQnt ? max : lengthQnt;
  }
});

rl.on('close', () => {
  console.log(res);
  process.exit(0);
});
