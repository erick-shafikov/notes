export default {};

const obj = {
    a: {
        c: () => {
            return 'x';
        },
    },
    e: {
        f: (x: boolean) => {
            return 5;
        },
    },
    g: {
        h: { i: () => ({ x: 'x' }) },
    },
};

type ReqDeepest<
    T extends any,
    K extends keyof T = keyof T
> = T[K] extends Function ? T[K] : ReqDeepest<T[K]>;

type Flatten<T> = {
    [K in keyof T as T[K] extends Function
        ? K
        : keyof Flatten<T[K]>]: ReqDeepest<T[K]>;
};

type Y = typeof obj;
type F = Flatten<Y>;
let f: F = {
    c: () => {
        return 'x';
    },
    f: (x: boolean) => {
        return 5;
    },
    i: () => ({
        x: 'x',
    }),
};
