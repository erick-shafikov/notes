//SSG страница
import LayoutPost from '@/components/layout-post';
import { TPost, fetchPosts, getSinglePost } from '@/services/api';
import { useRouter } from 'next/router';

import { ReactElement } from 'react';

function Page({ post }: { post: TPost }) {
  const router = useRouter();

  // If the page is not yet generated, this will be displayed
  // initially until getStaticProps() finishes running
  if (router.isFallback) {
    return <div>Loading...</div>;
  }

  return (
    <>
      <h1>Post: {post.title}</h1>
      <p>Post: {post.body}</p>
    </>
  );
}

Page.getLayout = function getLayout(page: ReactElement) {
  return <LayoutPost>{page}</LayoutPost>;
};

export default Page;

// вызов в build
export async function getStaticPaths() {
  const posts = await fetchPosts();

  // формируем пути для динамических роутов
  const paths = posts.map((post) => ({
    params: { id: `${post.id}` },
  }));

  // без пре-рендера путей
  return { paths: [], fallback: true };

  // Пре-рендер будет при build { fallback: false } другие пути вернут 404
  // с пре-рендером путей
  // return { paths, fallback: false };
  // fallback : false  - для несуществующей страницы выдаст 404
  // fallback : true  - вернет все html c пре рендером,
  // - не выдаст 404, при несуществующей странице
  // fallback: 'blocking'
}

// вызов в build
export async function getStaticProps({ params }: { params: { id: string } }) {
  //   Параметры - params, preview, draftMode, locales, defaultLocale, params contains the post `id`.
  const post = await getSinglePost(params.id);

  // Pass post data to the page via props
  return { props: { post } };
}
