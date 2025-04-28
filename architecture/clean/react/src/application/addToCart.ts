import { Product } from "../domain/product";
import { hasAllergy, User } from "../domain/user";
import { addProduct } from "../domain/cart";

import { CartStorageService, NotificationService } from "./ports";
import { useCartStorage } from "../services/storageAdapter";
import { useNotifier } from "../services/notificationAdapter";

//добавление в карту
export function useAddToCart() {
  //
  //опираемся нв интерфейс CartStorageService, а useCartStorage (возвращает контекст)
  const storage: CartStorageService = useCartStorage();
  //опираемся нв сервис useCartStorage (возвращает контекст)
  const notifier: NotificationService = useNotifier();

  function addToCart(user: User, product: Product): void {
    const warning = "This cookie is dangerous to your health! 😱";
    const isDangerous = product.toppings.some((item) => hasAllergy(user, item));
    if (isDangerous) return notifier.notify(warning);

    const { cart } = storage;
    const updated = addProduct(cart, product);
    storage.updateCart(updated);
  }

  return { addToCart };
}
