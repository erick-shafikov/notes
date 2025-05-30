//описание заказа
//импортируем currentDatetime так как это утилита
import { currentDatetime } from "../lib/datetime";
//можно импортировать из соседних сущностей нужные данные
import { totalPrice } from "./product";
import { Cart } from "./cart";
import { User } from "./user";

export type OrderStatus = "new" | "delivery" | "completed";

export type Order = {
  user: UniqueId;
  cart: Cart;
  created: DateTimeString;
  status: OrderStatus;
  total: PriceCents;
};

export function createOrder(user: User, cart: Cart): Order {
  return {
    cart,
    user: user.id,
    status: "new",
    created: currentDatetime(),
    total: totalPrice(cart.products),
  };
}
