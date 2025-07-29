# this

this теряется при использовании методов в других объектах и наследовании

```ts
class PaymentLoseContext {
  private data: Date = new Date();
  getDate() {
    return this.data;
  }
}
const newPayment1 = new PaymentLoseContext();

const user1 = {
  id: 1,
  paymentData: newPayment1.getDate(), //потеря контекста
};

class PaymentWithContext {
  private data: Date = new Date();
  getDate(this: PaymentWithContext) {
    //привязка контекста через ключевое this
    return this.data;
  }
  getDateArrow = () => {
    return this.data; //второй вариант привязка с помощью стрелочной функции
  };
}

const newPayment2 = new PaymentWithContext();

const user2 = {
  id: 1,
  paymentData: newPayment2.getDate(), //
  paymentData2: newPayment2.getDate.bind(newPayment2), //привязываем с помощью Bind
};

class brokeArrowLogic extends PaymentWithContext {
  save() {
    return super.getDateArrow(); //Не будет работать
    return this.getDateArrow(); //будет работать
  }
}
```
