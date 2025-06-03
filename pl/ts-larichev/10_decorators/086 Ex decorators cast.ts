export {};

function CratedAt<T extends { new (...args: any[]): {} }>(constructor: T) {
    return class extends constructor {
        createdAt = new Date();
    };
}

interface IUserService {
    users: number;
    getUserIDInDB(): number;
}

type CreatedAt = {
    //для идентификации нового свойства
    createdAt: Date;
};
@CratedAt
class UserService implements IUserService {
    users: number = 10000;

    getUserIDInDB(): number {
        return this.users;
    }
}

console.log((new UserService() as IUserService & CreatedAt).createdAt); //теперь можно обратиться
