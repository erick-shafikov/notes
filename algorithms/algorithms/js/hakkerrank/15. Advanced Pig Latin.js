function translate(sentence) {
  return sentence
    .split(" ")
    .map((item) => {
      return item.endsWith("ay") ? item : consonantReplacer(item);
    })
    .join(" ");
}

function consonantReplacer(str) {
  let fristIsWovel = true;
  function Replacer(str) {
    if (!!str.match(/\b[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]\w*\W*/)) {
      str = str.replace(
        /(\b[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])(\w+)(\W{0,})/,
        (match, firstletter, end, punctuation) => {
          return firstletter === firstletter.toLowerCase()
            ? `${end}${firstletter.toLowerCase()}` + punctuation
            : `${end.replace(
                /(\w)(\w+)/,
                (match, firstChar, lastChars) =>
                  firstChar.toUpperCase() + lastChars
              )}${firstletter.toLowerCase()}` + punctuation;
        }
      );
      fristIsWovel = false;
      return (str = Replacer(str));
    } else {
      return str.replace(/(\w*)(\W*)/i, (match, finalWord, punctuation) => {
        return fristIsWovel
          ? finalWord + "way" + punctuation
          : finalWord + "ay" + punctuation;
      });
    }
  }

  return Replacer(str);
}
alert(translate("hello")); // 'ellohay', 'failed to translate hello'
alert(translate("hello world")); // 'ellohay orldway', 'failed to translate hello world'
alert(translate("Hello World")); // 'Ellohay Orldway', 'failed to retain capital letters'
alert(translate("Pizza? Yes Please!!")); // 'Izzapay? Esyay Easeplay!!', 'failed to move punctuation'
alert(translate("How are you?")); //'Owhay areway ouyay?', 'failed to translate "How are you?"
