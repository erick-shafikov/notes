const readline = require("readline");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let input = [];
let res = -1;

rl.on("line", (inputItem) => {
  input.push(inputItem);
  if (input.length === 2) {
    const [length, string] = input;
    const set = new Set();
    let tempGoodString = "";
    let goodString = "";
    const stringArr = string.split("");

    for (let i = 0; i < length; i++) {
      for (let j = i; j < length; j++) {
        if (!set.has(stringArr[j])) {
          set.add(stringArr[j]);
          tempGoodString += stringArr[j];
          if (set.size === 4) {
            if (!goodString) goodString = tempGoodString;
            goodString =
              tempGoodString.length < goodString.length
                ? tempGoodString
                : goodString;
            set.clear();
            tempGoodString = "";
          }
        } else {
          tempGoodString += stringArr[j];
        }
      }

      tempGoodString = "";
      set.clear();
    }

    res = goodString.length || -1;
  }
});

rl.on("close", () => {
  console.log(res);
  process.exit(0);
});
