const linkedList5 = { value: 5, next: null };
const linkedList4 = { value: 4, next: linkedList5 };
const linkedList3 = { value: 3, next: linkedList4 };
const linkedList2 = { value: 2, next: linkedList3 };
const linkedList1 = { value: 1, next: linkedList2 };

const printList = (head) => {
  let current = head;

  while (true) {
    console.log(current);
    if (current.next) {
      current = current.next;
    } else {
      return;
    }
  }
};

const exclude = (head, numberToExclude) => {
  let prev = null;
  let current = head;

  while (true) {
    if (current.value === numberToExclude) {
      if (prev) {
        prev.next = current.next;
      }
    }
    if (current.next) {
      prev = current;
      current = current.next;
    } else {
      return;
    }
  }
};

exclude(linkedList1, 1);

console.log(printList(linkedList1));
