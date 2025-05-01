import { Product } from "../domain/product";
import { hasAllergy, User } from "../domain/user";
import { addProduct } from "../domain/cart";

import { CartStorageService, NotificationService } from "./ports";
import { useCartStorage } from "../services/storageAdapter";
import { useNotifier } from "../services/notificationAdapter";

//добавление в карту, название файла addToCart, экспортируем хук, так как использует контекст
export function useAddToCart() {
  //опираемся на интерфейс CartStorageService - это порт, а useCartStorage (возвращает контекст)
  const storage: CartStorageService = useCartStorage();
  //опираемся на сервис useCartStorage (возвращает контекст), NotificationService - это порт
  const notifier: NotificationService = useNotifier();

  // основной функционал - addToCart
  function addToCart(user: User, product: Product): void {
    const warning = "This cookie is dangerous to your health! 😱";
    // логика на проверку из domain
    const isDangerous = product.toppings.some((item) => hasAllergy(user, item));
    if (isDangerous) return notifier.notify(warning);
    //логика добавление в карту
    const { cart } = storage;
    const updated = addProduct(cart, product);
    storage.updateCart(updated);
  }

  return { addToCart };
}
