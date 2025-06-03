// Пример layout для каждый из страницы
import type { ReactElement } from 'react';
import Layout from '@/components/layout-main';

import type { NextPageWithLayout } from './_app';

const Page: NextPageWithLayout = () => {
  return <p>Main</p>;
};

Page.getLayout = function getLayout(page: ReactElement) {
  return <Layout>{page}</Layout>;
};

export default Page;
