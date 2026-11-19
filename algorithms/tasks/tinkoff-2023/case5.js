const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let inputArr = [];
let res;

rl.on('line', (inputItem) => {
  inputArr.push(inputItem);

  if (inputArr.length === 2) {
    let set = new Set();

    const [length, arr] = inputArr;

    const numbers = arr.split(' ').map((number) => +number);

    for (let i = 0; i < length; i++) {
      let current = numbers[i];
      let string = `${current},`;

      for (let j = i + 1; j < length; j++) {
        current += numbers[j];
        string += `${numbers[j]},`;
        if (current === 0) {
          set.add(string);

          for (let k = j + 1; k < length; k++) {
            string += `${numbers[k]},`;
            set.add(string);
          }
        }
      }
    }

    res = set.size;
  }
});

rl.on('close', () => {
  console.log(res);
  process.exit(0);
});
