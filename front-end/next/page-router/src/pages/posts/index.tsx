// Incremental Static Regeneration (ISR)
import { TPost, fetchPosts } from '@/services/api';
import { GetServerSideProps } from 'next';
import Link from 'next/link';

export default function Post({ posts }: { posts: TPost[] }) {
  return (
    <>
      {posts.map((post) => {
        return (
          <div key={post.id}>
            {/* <Link href={`post/${post.id}`}>{post.title}</Link> */}
            <Link
              href={{
                pathname: '/posts/[id]',
                query: { id: post.id },
              }}
            >
              {post.title}
            </Link>
          </div>
        );
      })}
    </>
  );
}

// не может быть экспортирована из не страничных фалов _app, _document, or _error.
export const getServerSideProps: GetServerSideProps<{
  posts: TPost[];
}> = async (context) => {
  const posts = await fetchPosts();

  //----------------------------- объект на context (параметр) ----------------------------

  // console.log('params:', context.params); //параметры строки
  // console.log('req:', context.req); //параметры запроса
  /* 
  req: {
    url: '/posts?__nextDataReq=1',
    method: 'GET',
    headers: [Object: null prototype] {
      'x-nextjs-data': '1',
      'x-forwarded-for': '::',
      'x-forwarded-host': 'localhost:3000',
      'x-forwarded-port': '3000',
      'x-forwarded-proto': 'http'
    }
  }
  */
  //параметры ответа
  // console.log('res:', context.res);
  /* 
    res: ServerResponseShim {
    headersSent: false,
    req: {
      url: '/posts?__nextDataReq=1',
      method: 'GET',
      headers: [Object: null prototype] {
        'x-nextjs-data': '1',
        'x-forwarded-for': '::',
        'x-forwarded-host': 'localhost:3000',
        'x-forwarded-port': '3000',
        'x-forwarded-proto': 'http'
      }
    }
  }
  */
  //параметры строки учитывая динамические роуты
  // console.log('query:', context.query); // query: { __nextDataReq: '1' }
  //нормализованная строка
  // console.log('resolvedUrl:', context.resolvedUrl); // /posts?__nextDataReq=1
  // локаль
  // console.log('locale:', context.locale, context.locales); // undefined undefined

  //----------------------------- объект на возврат ----------------------------

  //not found вариант вернет 404
  /* if (!posts) {
    return {
      notFound: true,
    };
  }*/

  //redirect - вариант с последующим перенаправлением
  /* if (!posts) {
    return {
      redirect: {
        destination: '/',
        permanent: false,
        statusCode: 307,
      },
    };
  } */

  return {
    props: {
      posts,
    },
  };
};
