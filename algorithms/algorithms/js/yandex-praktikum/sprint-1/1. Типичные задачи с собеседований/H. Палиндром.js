/**
 * @param {string} firstLetter
 * @param {string} secondLetter
 *
 * @returns {boolean}
 */
const isTheSameLetter = (firstLetter, secondLetter) =>
  firstLetter.toLowerCase() === secondLetter.toLowerCase();

/**
 * @param {string} letter
 *
 * @returns {boolean}
 */
const isLetter = (letter) => letter.toUpperCase() !== letter.toLowerCase();

/**
 * @param {string} string
 *
 * @returns {boolean}
 */
const isPalindrome = (string) => {
  let start = 0;
  let end = string.length - 1;
  let isPalindrome = true;

  while (start < end) {
    if (isLetter(string[start])) {
      if (isLetter(string[end])) {
        if (isTheSameLetter(string[start], string[end])) {
          start++;
          end--;
        } else {
          return false;
        }
      } else {
        end--;
      }
    } else {
      start++;
    }
  }

  return isPalindrome;
};

// console.log(isPalindrome("A man, a plan, a canal: Panama"));
// console.log(isPalindrome("xo"));
// console.log(isPalindrome("xox"));
console.log(isPalindrome("bacab"));
