class Stack {
  stack = [];
  max = -Infinity;

  push(newValue) {
    if (newValue > this.max) this.max = newValue;

    this.stack.push(newValue);
  }

  pop() {
    return this.stack.pop();
  }

  getMax() {
    return this.max;
  }
}

const stack = new Stack();

stack.push(1);
stack.push(2);
stack.push(3);
stack.pop();

console.log(stack.stack);
