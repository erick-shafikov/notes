import type { ITransaction } from '@/src/entities/models/transaction.interface';
//адаптер для сервиса
export interface ITransactionManagerService {
  startTransaction<T>(
    clb: (tx: ITransaction) => Promise<T>,
    parent?: ITransaction
  ): Promise<T>;
}
