function findMissingNumber(arr) {
  const n = arr.length + 1;
  const sumOfFirstN = (n * (n + 1)) / 2;

  let sumOfArray = 0;
  for (let i = 0; i < n - 1; i++) {
    sumOfArray = sumOfArray + arr[i];
  }

  let missingNumber = sumOfFirstN - sumOfArray;

  return missingNumber;
}

const arr1 = [1, 2, 5, 4, 6, 8, 7];
const missingNumber1 = findMissingNumber(arr1);
console.log("Missing Number: ", missingNumber);

function findMissing(arr, N) {
  let i;

  // Create an Array of size N
  // and filled with 0
  let temp = new Array(N).fill(0);

  // If array element exist then
  // set the frequency to 1
  for (i = 0; i < N; i++) {
    temp[arr[i] - 1] = 1;
  }

  let ans = 0;
  for (i = 0; i <= N; i++) {
    if (temp[i] === 0) ans = i + 1;
  }
  console.log(ans);
}

// Driver code
let arr2 = [1, 3, 7, 5, 6, 2];
let n = arr2.length;

// Function call
findMissing(arr2, n);

function findMissingNumber(arr) {
  arr.sort((a, b) => a - b);
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] !== i + 1) {
      return i + 1;
    }
  }
  return arr.length + 1;
}

const numbers = [1, 2, 3, 4, 5, 6, 8, 9, 10];
const missingNumber3 = findMissingNumber(numbers);
console.log("The missing number is:", missingNumber3);

// XOR решение

function findMissingNumber(arr) {
  const n = arr.length + 1;

  // Step 1: Calculate XOR of all numbers from 1 to N
  let xor_all = 0;
  for (let i = 1; i <= n; i++) {
    xor_all ^= i;
  }

  // Step 2: Calculate XOR of all elements in the array
  let xor_arr = 0;
  for (let i = 0; i < arr.length; i++) {
    xor_arr ^= arr[i];
  }

  // Step 3: XOR of xor_all and xor_arr gives the missing number
  const missingNumber = xor_all ^ xor_arr;

  return missingNumber;
}

// Test case
const ar4 = [1, 2, 3, 5];
const missingNumber = findMissingNumber(arr4);
console.log("Missing Number: ", missingNumber);

const arr5 = [1, 4, 3, 2, 6, 5, 7, 10, 9];
const missingNumber2 = findMissingNumber(arr5);
console.log("Missing Number: ", missingNumber2);
