"use strict";
/* export {};

function useState<T>(initial: T): [T, (newValue: T) => void] {
    let value = initial;

    function setValue(newValue: T) {
        value = newValue;
    }

    return [value, setValue];
}

const [a, setA] = useState(1);
const [b, setB] = useState('1'); */
//-----------------------------------------------------------------
function insertNestedKeys(obj1, obj2) {
    for (let key in obj2) {
        if (typeof obj2[key] === 'object') {
            insertNestedKeys(obj1[key], obj2[key]);
        }
        else {
            obj1[key] = obj2[key];
        }
    }
}
function useObjState(initial) {
    let value = initial;
    function setValue(newValue) {
        insertNestedKeys(value, newValue);
    }
    return [value, setValue];
}
const [c, setC] = useObjState({
    id: 1,
    name: 'Name',
    role: 'admin',
    auth: {
        token: 'qwerty',
        exp: 1000,
        data: {
            last: '10.10.10',
            update: '19.19.19',
            lastChanges: ['1.1.1', '2.2.2'],
        },
    },
});
setC({
    name: 'some',
    auth: {
        exp: 100,
        data: {
            last: '11.11.11',
            lastChanges: ['3.3.3', '4.4.4'],
        },
    },
});
console.log(c);
