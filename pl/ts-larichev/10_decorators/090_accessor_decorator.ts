export {};

interface IUserService {
    users: number;
    getUsersInDatabase(): number;
}

class UserService implements IUserService {
    private _users: number = 1000;

    @Log()
    set users(num: number) {
        this._users = num;
    }

    // @Log() - нельзя устанавливать на set и get сразу , так как отрабатывает 2 раза и на get и на set
    get users() {
        return this._users;
    }

    getUsersInDatabase(): number {
        throw new Error('Error');
    }
}
//дескриптор на акссесоры get и set
function Log() {
    return (
        target: Object, //UserService
        _: string | symbol, //
        descriptor: PropertyDescriptor
        /* 
        descriptor { 
        configurable?: boolean;
        enumerable?: boolean;
        value?: any;
        writable?: boolean;
        get?(): any;
        set?(v: any): void;
    }
     */
    ) => {
        const set = descriptor.set;

        descriptor.set = (...args: any) => {
            console.log(...args);
            set?.apply(target, args);
        };
    };
}

console.log(new UserService().getUsersInDatabase());
