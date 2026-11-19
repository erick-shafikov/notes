const removeX = (str) => {
  return str.split("").filter((letter) => letter !== "X");
};

const indexesOfLetter = (arr, letter) => {
  return arr
    .map((item, index) => {
      if (item === letter) return index;
    })
    .filter((item) => item !== undefined);
};

const canTransform = function (start, end) {
  const startOmitX = removeX(start);
  const endOmitX = removeX(end);

  if (startOmitX.join("") !== endOmitX.join("")) return false;

  const indexesOfLInStart = indexesOfLetter([...start], "L");

  const indexesOfLInEnd = indexesOfLetter([...end], "L");

  const indexesOfRInStart = indexesOfLetter([...start], "R");

  const indexesOfRInEnd = indexesOfLetter([...end], "R");

  for (let i = 0; i < indexesOfLInStart.length; i++) {
    if (indexesOfLInStart[i] < indexesOfLInEnd[i]) return false;
  }

  for (let i = 0; i < indexesOfRInStart.length; i++) {
    if (indexesOfRInStart[i] > indexesOfRInEnd[i]) return false;
  }

  return true;
};

// console.log(canTransform("LXXLXRLXXL", "XLLXRXLXLX"));

console.log(canTransform("XLLR", "LXLX"));
console.log(canTransform("XXXXXLXXXX", "LXXXXXXXXX"));
