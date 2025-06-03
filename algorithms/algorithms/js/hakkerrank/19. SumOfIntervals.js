function sumIntervals(intervals) {
  if (intervals.length === 1) return intervals[0][1] - intervals[0][0];

  intervals.sort((a, b) => a[0] - b[0]);

  let finalIntervals = [intervals[0]];
  let current = 0;
  for (let i = 1; i < intervals.length; i++) {
    if (
      finalIntervals[current][0] <= intervals[i][0] &&
      intervals[i][0] <= finalIntervals[current][1]
    ) {
      finalIntervals[current][1] = Math.max(
        finalIntervals[current][1],
        intervals[i][1]
      );
    } else {
      finalIntervals.push(intervals[i]);
      current++;
    }
  }

  return finalIntervals.reduce(
    (prev, current) => prev + current[1] - current[0],
    0
  );
}

console.log(`${sumIntervals([[1, 5]])}`);
console.log(
  `${sumIntervals([
    [1, 5],
    [6, 10],
  ])}`
);
console.log(
  `${sumIntervals([
    [1, 5],
    [1, 5],
  ])}`
);
console.log(
  `${sumIntervals([
    [1, 4],
    [7, 10],
    [3, 5],
  ])}`
);
console.log(
  `${sumIntervals([
    [1, 5],
    [10, 20],
    [1, 6],
    [16, 19],
    [5, 11],
  ])}`
); //[[1,5],[1, 6],[5, 11],[10, 20],[16, 19],]
console.log(
  `${sumIntervals([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
  ])}`
);
console.log(
  `${sumIntervals([
    [-100, -5],
    [1, 2],
    [1, 7],
  ])}`
);
