function solution(list) {
  let str = "";

  for (let i = 0; i < list.length; i++) {
    if (list[i] == list[i + 1] - 1 && list[i + 1] == list[i + 2] - 1) {
      for (let j = i; ; j++) {
        if (list[j] !== list[j + 1] - 1) {
          str += `${list[i]}-${list[j]}, `;
          i = j;
          break;
        } else {
          continue;
        }
      }
    } else {
      str += list[i] + ",";
    }
  }

  return str;
}

alert(
  solution([
    -6, -3, -2, -1, 0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 17, 18, 19, 20,
  ])
);
