// const howMuchNumbersOneInBinary = (number) =>
//   number.toString(2).replace(/0/g, "").length;

const howMuchNumbersOneInBinary = (number) => {
  count = 0;
  while (number != 0) {
    number = number & (number - 1);
    count++;
  }

  return count;
};

console.log(howMuchNumbersOneInBinary(5));
