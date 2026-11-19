// const excludeZero = (string) =>
//   string
//     .split(",")
//     .filter((number) => number !== "0")
//     .join(",");

const excludeZero = (string) => {
  for (let i = 0; i < string.length; i++) {
    if (string[i] === "0") {
      string = string.slice(0, i - 1) + string.slice(i + 1, string.length);
      i--;
    }
  }

  return string;
};

console.log(excludeZero("1,2,0,3,0,0,4,5,6,7,0,0,0,0,8"));
