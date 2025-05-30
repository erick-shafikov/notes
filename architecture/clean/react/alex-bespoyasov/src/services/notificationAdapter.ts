import { NotificationService } from "../application/ports";
//диалоговое окно
export function useNotifier(): NotificationService {
  return {
    notify: (message: string) => window.alert(message),
  };
}
