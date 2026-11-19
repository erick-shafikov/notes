function productFib(prod) {
  let num1 = 0;
  let num2 = 1;
  let num3 = 0;

  while (true) {
    num3 = num1 + num2;

    if (num1 * num2 < prod && prod < num2 * num3) return [num2, num3, false];
    if (num1 * num2 == prod) return [num1, num2, true];

    num1 = num2;
    num2 = num3;
  }
}

//alert(productFib(4895)) //[55, 89, true]
//alert(productFib(5895)) //[89, 144, false]
alert(productFib(74049690)); //[6765, 10946, true]
//productFib(84049690) //[10946, 17711, false]
//productFib(193864606) //[10946, 17711, true]
//productFib(447577) //[610, 987, false]
//productFib(602070) //[610, 987, true]
