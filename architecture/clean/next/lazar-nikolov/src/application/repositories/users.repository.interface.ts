import type { User, CreateUser } from '@/src/entities/models/user';
import type { ITransaction } from '@/src/entities/models/transaction.interface';
//адаптер для репозитория - описывает интерфейс взаимодействия
export interface IUsersRepository {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(input: CreateUser, tx?: ITransaction): Promise<User>;
}
