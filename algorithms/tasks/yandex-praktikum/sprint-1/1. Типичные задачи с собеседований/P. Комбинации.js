const buttons = {
  2: ["a", "b", "c"],
  3: ["d", "e", "f"],
  4: ["g", "h", "i"],
  5: ["k", "l", "m"],
  6: ["n", "o", "p"],
  7: ["q", "r", "s"],
  8: ["t", "u", "v"],
  9: ["w", "x", "y", "z"],
};

/**
 * @param {string} string
 *
 * @returns {string}
 */
const createCombinations = (string) => {
  let step = 1;
  const stringLength = string.length;
  let combinations = [...buttons[string[0]]];

  while (step < stringLength) {
    const currentCombinationLength = combinations.length;

    for (let i = 0; i < currentCombinationLength; i++) {
      if (combinations[i] !== null) {
        const currentButtonLength = buttons[string[step]].length;

        for (let j = 0; j < currentButtonLength; j++) {
          combinations.push(`${combinations[i]}${buttons[string[step]][j]}`);
        }
      }

      combinations[i] = null;
    }

    step++;
  }

  return combinations.filter(Boolean).join(" ");
};

console.log(createCombinations("234"));
