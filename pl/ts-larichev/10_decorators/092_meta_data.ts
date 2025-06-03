import 'reflect-metadata';
export {};

const POSITIVE_METADATA_KEY = Symbol('POSITIVE_METADATA_KEY');

interface IUserService {
    users: number;
    getUsersInDatabase(): number;
}

class UserService implements IUserService {
    users: number;

    private _users: number = 1000;

    getUsersInDatabase(): number {
        return this._users;
    }

    @Validate()
    setUsersInDatabase(@Positive() num: number): void {
        this._users = num;
    }
}
//дескриптор на парметры
function Positive() {
    return (
        target: Object, //UserService
        propertyKey: string | symbol, //setUsersInDatabase
        parameterIndex: number // Индекс параметра
    ) => {
        console.log(Reflect.getOwnMetadata('design:type', target, propertyKey)); //[Function: Function]
        console.log(
            Reflect.getOwnMetadata('design:paramtypes', target, propertyKey)
        ); //undefined
        console.log(
            Reflect.getOwnMetadata('design:returntype', target, propertyKey)
        ); //1000

        let existParams: number[] =
            Reflect.getOwnMetadata(
                POSITIVE_METADATA_KEY,
                target,
                propertyKey
            ) || [];

        existParams.push(parameterIndex);
        Reflect.defineMetadata(
            POSITIVE_METADATA_KEY,
            existParams,
            target,
            propertyKey
        );
    };
}

function Validate() {
    return (
        target: Object,
        propertyKey: string | symbol,
        descriptor: TypedPropertyDescriptor<(...args: any[]) => any>
    ) => {
        let method = descriptor.value;

        descriptor.value = function (...args: any[]) {
            let positiveParams: number[] = Reflect.getOwnMetadata(
                POSITIVE_METADATA_KEY,
                target,
                propertyKey
            );
            if (positiveParams) {
                for (let index of positiveParams) {
                    if (args[index] < 0) {
                        throw new Error('Number has to be more than 0');
                    }
                }
            }

            return method?.apply(this, args);
        };
    };
}
console.log(new UserService().setUsersInDatabase(-1));
