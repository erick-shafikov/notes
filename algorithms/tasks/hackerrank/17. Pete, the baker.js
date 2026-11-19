function cakes(recipe, available) {
  let arr = [];
  for (ing in recipe) {
    if (ing in available && recipe[ing] <= available[ing]) {
      arr.push(Math.floor(available[ing] / recipe[ing]));
    } else {
      return 0;
    }
  }
  return Math.min(...arr) === 0 ? 0 : Math.min(...arr);
}

console.log(
  cakes(
    { flour: 500, sugar: 200, eggs: 1 },
    { flour: 1200, sugar: 1200, eggs: 5, milk: 200 }
  )
);
console.log(
  cakes(
    { apples: 3, flour: 300, sugar: 150, milk: 100, oil: 100 },
    { sugar: 500, flour: 2000, milk: 2000 }
  )
);
