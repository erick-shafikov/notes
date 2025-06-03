arr = [3, 2, 1, 3];

function BirthdayCakeCandles(arr) {
  max = Math.max(...arr);
  let i = 0;
  arr.map((item) => {
    if (item == max) {
      i++;
    }
  });
  return i;
}

alert(BirthdayCakeCandles(arr));
