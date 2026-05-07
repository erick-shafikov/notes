# Intl

## статические методы

## getCanonicalLocales()

⇒ массив, содержащий канонические названия локалей

```js
console.log(Intl.getCanonicalLocales("EN-US"));
// Expected output: Array ["en-US"]

console.log(Intl.getCanonicalLocales(["EN-US", "Fr"]));
// Expected output: Array ["en-US", "fr"]

try {
  Intl.getCanonicalLocales("EN_US");
} catch (err) {
  console.log(err.toString());
  // Expected output: RangeError: invalid language tag: "EN_US"
}
```

## supportedValuesOf()

⇒ массив, содержащий поддерживаемые календари, кодировки, валюты, системы счисления или значения единиц измерения, поддерживаемые реализацией.

```js
//вернет все значения для типов календарей "buddhist", "chinese", "coptic", "dangi", "ethioaa", "ethiopic", "gregory", "hebrew", "indian", "islamic", "islamic-civil", "islamic-rgsa", "islamic-tbla", "islamic-umalqura", "iso8601", "japanese", "persian", "roc"
Intl.supportedValuesOf("calendar");
// "compat", "emoji", "eor", "phonebk", "pinyin", "searchjl", "stroke", "trad", "unihan", "zhuyin"
Intl.supportedValuesOf("collation");
// "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN", "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY", "COP", "CRC", "CUC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG", "HUF", "IDR", "ILS", "INR", "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF", "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD", "SHP", "SLL", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB", "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX", "USD", "UYU", "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XDR", "XOF", "XPF", "XSU", "YER", "ZAR", "ZMW", "ZWL"
Intl.supportedValuesOf("currency");
// "adlm", "ahom", "arab", "arabext", "bali", "beng", "bhks", "brah", "cakm", "cham", "deva", "diak", "fullwide", "gara", "gong", "gonm", "gujr", "gukh", "guru", "hanidec", "hmng", "hmnp", "java", "kali", "kawi", "khmr", "knda", "krai", "lana", "lanatham", "laoo", "latn", "lepc", "limb", "mathbold", "mathdbl", "mathmono", "mathsanb", "mathsans", "mlym", "modi", "mong", "mroo", "mtei", "mymr", "mymrepka", "mymrpao", "mymrshan", "mymrtlng", "nagm", "newa", "nkoo", "olck", "onao", "orya", "osma", "outlined", "rohg", "saur", "segment", "shrd", "sind", "sinh", "sora", "sund", "sunu", "takr", "talu", "tamldec", "telu", "thai", "tibt", "tirh", "tnsa", "vaii", "wara", "wcho"
Intl.supportedValuesOf("numberingSystem");
// "Africa/Abidjan"..."Pacific/Wallis"
Intl.supportedValuesOf("timeZone");
// "acre", "bit", "byte", "celsius", "centimeter", "day", "degree", "fahrenheit", "fluid-ounce", "foot", "gallon", "gigabit", "gigabyte", "gram", "hectare", "hour", "inch", "kilobit", "kilobyte", "kilogram", "kilometer", "liter", "megabit", "megabyte", "meter", "microsecond", "mile", "mile-scandinavian", "milliliter", "millimeter", "millisecond", "minute", "month", "nanosecond", "ounce", "percent", "petabyte", "pound", "second", "stone", "terabit", "terabyte", "week", "yard", "year"
Intl.supportedValuesOf("unit");
```

# Intl.Collator()

## конструктор принимает

```ts
new Intl.Collator(locales, options);

type Locales = "co" | "kn" | "kf";
type Options = {
  usage: "sort" | "search";
  localeMatcher: "lookup" | "best fit";
  collation: "emoji" | "pinyin" | "stroke";
  numeric: boolean;
  caseFirst: "upper" | "lower" | "false";
  sensitivity: "base" | "accent" | "case" | "variant";
  ignorePunctuation: boolean;
};
```

## статические свойства

### supportedLocalesOf

⇒ массив, содержащий те языковые локали, которые поддерживаются в настройках сортировки без необходимости использования локали по умолчанию среды выполнения.

```ts
Intl.Collator.supportedLocalesOf(locales, options);

type Locales = string; // BCP 47 language tag
type Options = { localeMatcher: "lookup" | "best fit" };
```

```js
const locales = ["ban", "id-u-co-pinyin", "de-ID"];
const options = { localeMatcher: "lookup" };
console.log(Intl.Collator.supportedLocalesOf(locales, options));
// ["id-u-co-pinyin", "de-ID"]
```

## методы экземпляра

### compare

Сравнивает две строки в соответствии с порядком сортировки данного объекта-сопоставителя.

```js
const enCollator = new Intl.Collator("en");
const deCollator = new Intl.Collator("de");
const svCollator = new Intl.Collator("sv");

console.log(enCollator.compare("z", "a") > 0);
// Expected output: true

console.log(deCollator.compare("z", "ä") > 0);
// Expected output: true

console.log(svCollator.compare("z", "ä") > 0);
// Expected output: false
```

### resolvedOptions

⇒ новый объект со свойствами, отражающими параметры, вычисленные во время инициализации этого объекта Collator.

```ts
const numberDe = new Intl.NumberFormat("de-DE");
const numberAr = new Intl.NumberFormat("ar");

const de = new Intl.Collator("de", { sensitivity: "base" });
const usedOptions: UsedOptions = de.resolvedOptions();

type UsedOptions = {
  locale: "de";
  usage: "sort";
  sensitivity: "base";
  ignorePunctuation: false;
  collation: "default";
  numeric: false;
};
```
