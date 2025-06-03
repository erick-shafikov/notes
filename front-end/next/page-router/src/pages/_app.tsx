//компонент из документации, обеспечивает layout'ы для конкретной страницы
import type { ReactElement, ReactNode } from 'react';
import type { NextPage } from 'next';
import type { AppProps } from 'next/app';
// глобальные стили
import '@/styles/global-styles.css';

export type NextPageWithLayout<P = {}, IP = P> = NextPage<P, IP> & {
  getLayout?: (page: ReactElement) => ReactNode;
};

type AppPropsWithLayout = AppProps & {
  Component: NextPageWithLayout;
};

//вариант где layout для каждой страницы определяется отдельно

/* export default function MyApp({ Component, pageProps }: AppPropsWithLayout) {
  // Use the layout defined at the page level, if available
  const getLayout = Component.getLayout ?? ((page) => page);
  
  return getLayout(<Component {...pageProps} />);
} */

//вариант layout для каждой страницы общий <Layout />

import Layout from '@/components/layout-main';
export default function MyApp({ Component, pageProps }: AppPropsWithLayout) {
  return (
    <Layout>
      <Component {...pageProps} />
    </Layout>
  );
}
