const excludeRepeating = (string) => {
  const numbers = string.split(" ");
  const set = new Set();

  for (let num in numbers) {
    number = numbers[num];

    if (set.has(number)) return number;

    set.add(number);
  }
};

console.log(excludeRepeating("1 12 13 3 4 5 6 12"));
