let arr = [-1, -2, -3, 1, 2, 3, 0, 0, 0];

function func(arr) {
  let positiveRatio = 0;
  let negativeRatio = 0;
  let zeroRatio = 0;

  arr.forEach((element) => {
    if (element < 0) {
      ++negativeRatio / arr.length;
    } else if (element > 0) {
      ++positiveRatio / arr.length;
    } else if (element == 0) {
      ++zeroRatio / arr.length;
    }
  });

  positiveRatio = positiveRatio.toFixed(6);
  negativeRatio = negativeRatio.toFixed(6);
  zeroRatio = zeroRatio.toFixed(6);

  return `${positiveRatio} ${negativeRatio} ${zeroRatio}`;
}

alert(func(arr));
alert(arr);
