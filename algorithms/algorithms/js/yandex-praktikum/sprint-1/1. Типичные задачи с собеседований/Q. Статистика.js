/**
 * @param {number[]} numbers
 *
 * @returns {number}
 */
const maxMultiplication = (numbers) => {
  numbers.sort((a, b) => a - b);

  const lessThanZero = [];
  const moreThanZero = [];
  const zeroCombinations = [];
  const numbersLength = numbers.length;

  for (let i = 0; i < numbersLength; i++) {
    if (numbers[i] > 0) {
      moreThanZero.push(numbers[i]);
    } else {
      lessThanZero.push(numbers[i]);
    }
  }

  const lessThanZeroLength = lessThanZero.length;
  const moreThanZeroLength = moreThanZero.length;

  for (let i = 0; i < lessThanZeroLength; i++) {
    let currentSum = lessThanZero[i];
    for (let j = i + 1; j < lessThanZeroLength; j++) {
      currentSum += lessThanZero[j];
      for (let k = 0; k < moreThanZeroLength; k++) {
        currentSum += moreThanZero[k];

        if (currentSum === 0)
          zeroCombinations.push([
            lessThanZero[i],
            lessThanZero[j],
            moreThanZero[k],
          ]);
      }
    }
  }

  for (let i = 0; i < moreThanZeroLength; i++) {
    let currentSum = moreThanZero[i];
    for (let j = i + 1; j < moreThanZeroLength; j++) {
      currentSum += moreThanZero[j];
      for (let k = 0; k < lessThanZeroLength; k++) {
        currentSum += lessThanZero[k];

        console.log(moreThanZero[i], lessThanZero[j], lessThanZero[k]);

        if (currentSum === 0)
          zeroCombinations.push([
            lessThanZero[i],
            lessThanZero[j],
            moreThanZero[k],
          ]);
      }
    }
  }

  const zeroCombinationsLength = zeroCombinations.length;
  let max = -Infinity;

  console.log(zeroCombinations);

  for (let i = 0; i < zeroCombinationsLength; i++) {
    const multiplication =
      zeroCombinations[i][0] * zeroCombinations[i][1] * zeroCombinations[i][2];

    console.log(zeroCombinations[0]);
    if (multiplication > max) {
      max = multiplication;
    }
  }

  return max;
};

// console.log(maxMultiplication([-2, -1, 3, -1, -2]));
console.log(maxMultiplication([-2, 1, 1, -1, -2]));
