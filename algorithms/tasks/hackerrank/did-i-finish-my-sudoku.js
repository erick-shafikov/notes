[
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 0, 3, 4, 9],
  [1, 0, 0, 3, 4, 2, 5, 6, 0],
  [8, 5, 9, 7, 6, 1, 0, 2, 0],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 0, 1, 5, 3, 7, 2, 1, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 0, 0, 4, 8, 1, 1, 7, 9],
];

function doneOrnot(board) {
  let isRowComplete = true;
  let isColumnComplete = true;
  let isSquareComplete = true;
  let sum = 0;

  for (let i = 0; i < 9; i++) {
    sum = 0;
    for (let j = 0; j < 9; j++) {
      sum += board[i][j];
    }
    if (sum != 45) isRowComplete = false;
  }

  for (let i = 0; i < 9; i++) {
    sum = 0;
    for (let j = 0; j < 9; j++) {
      sum += board[j][i];
    }
    if (sum != 45) isColumnComplete = false;
  }

  for (let k = 0; k < 3; k++) {
    for (let n = 0; n < 3; n++) {
      sum = 0;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          sum += board[i + 3 * k][j + 3 * n];
        }
      }
      alert(sum);
      if (sum != 45) isSquareComplete = false;
    }
  }

  return isColumnComplete && isRowComplete && isSquareComplete
    ? "Finished!"
    : "Try again!";
}

let res = doneOrnot([
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
]);

let res2 = doneOrnot([
  [5, 3, 4, 6, 7, 8, 9, 1, 2],
  [6, 7, 2, 1, 9, 5, 3, 4, 8],
  [1, 9, 8, 3, 4, 2, 5, 6, 7],
  [8, 5, 9, 7, 6, 1, 4, 2, 3],
  [4, 2, 6, 8, 5, 3, 7, 9, 1],
  [7, 1, 3, 9, 2, 4, 8, 5, 6],
  [9, 6, 1, 5, 3, 7, 2, 8, 4],
  [2, 8, 7, 4, 1, 9, 6, 3, 5],
  [3, 4, 5, 2, 8, 6, 1, 7, 9],
]);

alert(res);
alert(res2);
