import { PaymentService } from "../application/ports";
import { fakeApi } from "./api";
//вызов апи на выплату
export function usePayment(): PaymentService {
  return {
    tryPay(amount: PriceCents) {
      return fakeApi(true);
    },
  };
}
