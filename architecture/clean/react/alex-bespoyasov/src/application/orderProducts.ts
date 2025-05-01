import { User } from "../domain/user";
import { Cart } from "../domain/cart";
import { createOrder } from "../domain/order";

// интерфейсы портов находятся в application,
// но их реализация находится в services.
import { usePayment } from "../services/paymentAdapter";
import { useNotifier } from "../services/notificationAdapter";
import { useCartStorage, useOrdersStorage } from "../services/storageAdapter";

export function useOrderProducts() {
  const notifier = useNotifier();
  const payment = usePayment();
  const orderStorage = useOrdersStorage();
  const cartStorage = useCartStorage();

  async function orderProducts(user: User, cart: Cart) {
    const order = createOrder(user, cart);

    const paid = await payment.tryPay(order.total);
    if (!paid) return notifier.notify("The payment wasn't successful 🤷");

    const { orders } = orderStorage;
    orderStorage.updateOrders([...orders, order]);
    cartStorage.emptyCart();
  }

  return { orderProducts };
}
