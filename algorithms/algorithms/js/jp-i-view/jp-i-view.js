// case 1
const users = [
  {
    name: "John",
  },
];

const cloneUser = [...users];
const [firstUser] = users;
const [firsCloneUser] = users;

firsCloneUser.name = "Pete";

console.log(users === cloneUser); //false
console.log(firstUser === firsCloneUser); //true

//case2 написать функцию задержки между двумя логами

(async () => {
  const delay = async (ms) => {
    return new Promise((res, rej) => {
      setTimeout(() => {
        res();
      }, ms);
    });
  };

  console.log("begin");
  await delay(5000);
  console.log("after 5 sec");
})();

//case3

const a = {};
const b = {};

const obj = {
  [a]: 1,
  [b]: 2,
};

console.log(obj[a] + obj[b]); //
const map = new Map([
  [a, 1],
  [b, 2],
]);
console.log(map.get(a) + map.get(b)); //

//case4

(async () => {
  const task1 = () => {
    console.log("1");
    return Promise.resolve("2");
  };

  const task2 = () => {
    console.log("3");
    return Promise.reject("4");
  };

  try {
    const results = await Promise.all([task1(), task2()]);
  } catch (error) {
    console.log(error);
  } finally {
    console.log("5");
  } //1,3, 4, 5
})();

//case5

const getName = () => {
  console.log(this.username);
};

const obj1 = {
  username: "Pete",
  getName,
};

getName(); //undefined
obj1.getName(); //undefined
getName.bind(obj1)(); //undefined

// case6

const argConcat = (args) => {
  return args.length
    ? args.reduce((acc, cur) => {
        return `${acc}${cur}`;
      }, "")
    : "empty";
};

const memoized = (cb) => {
  const cache = {};

  return function (...args) {
    const argKey = argConcat(args);
    if (cache[argKey]) {
      console.log("from cache");
      return cache[argKey];
    } else {
      console.log("not from cache");
      const res = cb(...args);

      cache[argKey] = res;

      return res;
    }
  };
};

const memoizedSum = memoized((a, b) => {
  console.log(`calculate sum ${a} + ${b}`);

  return a + b;
});

console.log(memoizedSum(1, 2));
console.log(memoizedSum(3, 4));
console.log(memoizedSum(1, 2));
