function pigIt(str) {
  return str
    .split(" ")
    .map((item) =>
      item.replace(/(\w|\W)(\w*)/, (match, first, last) => {
        return first.match(/\W/)
          ? first
          : last.match(/\w*?ay/)
          ? match
          : last + first + "ay";
      })
    )
    .join(" ");
}
/* 
    function pigIt(str){
        return str.replace(/(\w)(\w*)(\s|$)/g, "\$2\$1ay\$3")
} */

alert(`INPUT : 'Pig latin is cool' OUTPUT: ${pigIt("Pig latin is cool")}`);
alert(`INPUT : 'This is my string' OUTPUT: ${pigIt("This is my string")}`);
alert(
  `INPUT : 'Oay emporatay oay oresmay !ay' OUTPUT: ${pigIt(
    "Oay emporatay oay oresmay !ay"
  )}`
);
alert(
  `INPUT: 'O emporatay o oresmay' !OUTPUT: ${pigIt("O emporatay o oresmay !")}`
);
