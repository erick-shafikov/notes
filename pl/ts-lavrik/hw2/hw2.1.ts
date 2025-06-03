export default {};

type TMethods = 'GET' | 'POST' | 'PUT' | 'PATCH';

type TOptions = {
    method?: TMethods;
    body?: BodyInit;
};

type TProduct = [{ name: string }];

type TSuccess<T> = {
    res: true;
    data: T;
};

type TError = {
    res: false;
    error: Error;
};

type TResponse<T> = TSuccess<T> | TError;

async function getJson<S>(
    url: string,
    options: TOptions = {}
): Promise<TResponse<S>> {
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        return { res: true, data };
    } catch (e) {
        return {
            res: false,
            error: e instanceof Error ? e : new Error('error'),
        };
    }
}

const a = getJson<TProduct>('www').then((res) => {
    if (res.res) {
        console.log(res);
    } else {
        res.error;
    }
});
