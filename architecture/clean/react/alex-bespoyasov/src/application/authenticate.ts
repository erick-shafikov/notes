import { UserName } from "../domain/user";
import { useAuth } from "../services/authAdapter";
import { useUserStorage } from "../services/storageAdapter";

// интерфейсы портов находятся в application,
// но их реализация находится в services.
import { AuthenticationService, UserStorageService } from "./ports";

export function useAuthenticate() {
  // Обычно мы получаем доступ к сервисам через внедрение зависимостей.
  // Здесь мы можем использовать хуки как кривые «DI-контейнеры».

  // Функция варианта использования не вызывает сторонние сервисы напрямую,
  // вместо этого она полагается на интерфейсы, которые мы объявили ранее.
  const storage: UserStorageService = useUserStorage();
  const auth: AuthenticationService = useAuth();

  // В идеале мы бы передали команду в качестве аргумента,
  // которая инкапсулировала бы все входные данные.
  async function authenticate(name: UserName, email: Email): Promise<void> {
    const user = await auth.auth(name, email);
    storage.updateUser(user);
  }

  return {
    user: storage.user,
    authenticate,
  };
}
