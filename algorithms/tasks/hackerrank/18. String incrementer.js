function incrementString(strng) {
  if (strng === "") return "1";
  return `${strng.match(/(\D*)(\d*)/)[1]}${incr(strng.match(/(\D*)(\d*)/)[2])}`;
}

function incr(str) {
  let plusOne = Number(str) + 1;
  return str.length < plusOne.toString().length
    ? plusOne
    : addNulls(str, plusOne.toString());
}

function addNulls(strLong, strShort) {
  let finalString = [...Array.from(strShort).reverse()];

  for (let i = 0; i < strLong.length - strShort.length; i++) {
    finalString.push(0);
  }

  return finalString.reverse().join("");
}

console.log(`${incrementString("foobar000")}, "foobar001"`);
console.log(`${incrementString("foo")}, "foo1"`);
console.log(`${incrementString("foobar001")}, "foobar002"`);
console.log(`${incrementString("foobar99")}, "foobar100"`);
console.log(`${incrementString("foobar099")}, "foobar100"`);
console.log(`${(incrementString(""), "1")}`);
