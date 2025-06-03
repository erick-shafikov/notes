// создать функцию проверки значения в объекте

function hasValue(obj, name){
    // code
}

const data = {
    name: "Ivan",
    address: {
        home: 4,
        n: 0
    }
}

console.log(hasValue(data, "name")); // true
console.log(hasValue(data, "address.n")); // true
console.log(hasValue(data, "address.street")); // false