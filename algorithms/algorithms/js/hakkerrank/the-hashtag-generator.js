function generateHashtag(str) {
  str = str
    .split(" ")
    .map((item) => {
      if (item === " ") return "";
      if (item.toString().length > 139) return false;
      return item.slice(0, 1).toUpperCase() + item.slice(1);
    })
    .join("");

  if (str.match("false") || str === "") return false;
  return `#${str}`;
}

alert(generateHashtag("")); // false, "Expected an empty string to return false")
alert(generateHashtag(" ".repeat(200))); //false, "Still an empty string")
alert(generateHashtag("Do We have A Hashtag")); // "#DoWeHaveAHashtag", "Expected a Hashtag (#) at the beginning.")
alert(generateHashtag("Codewars")); //"#Codewars", "Should handle a single word.")
alert(generateHashtag("Codewars Is Nice")); //"#CodewarsIsNice", "Should remove spaces.")
alert(generateHashtag("Codewars is nice")); // "#CodewarsIsNice", "Should capitalize first letters of words.")
alert(generateHashtag("code" + " ".repeat(140) + "wars")); // "#CodeWars")
alert(
  generateHashtag(
    "Looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong Cat"
  )
); // false, "Should return false if the final word is longer than 140 chars.")
alert(generateHashtag("a".repeat(139))); //"#A" + "a".repeat(138), "Should work")
alert(generateHashtag("a".repeat(140))); //false,
