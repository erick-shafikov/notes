(function () {
  // НОД для двух чисел - если есть остаток от деления, то проверить делится ли меньшее на этот остаток
  const euclideanForTwo = (a, b) => (!b ? a : euclideanForTwo(b, a % b));

  const euclideanForArr = (arr) => {
    let result = arr[0];
    let newArr = arr?.slice(1);

    newArr.map((item) => {
      result = euclideanForTwo(result, item);
    });

    return result;
  };

  // console.log(euclideanForTwo(75, 50));
  console.log(euclideanForArr([1680, 640, 3360, 160, 240, 168000]));
})();
