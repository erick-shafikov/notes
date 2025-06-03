import Layout from '@/components/layout-main';
import router from 'next/router';
import { ReactNode } from 'react';

async function testMiddleware() {
  const res = await fetch('/about/middleware-test');
  const json = await res.json();

  console.log('from middleware:', json);
}

function About() {
  testMiddleware();
  return (
    <>
      about Page
      <button onClick={() => router.push('/')}>Click to home</button>
      <button onClick={() => router.push('/posts')}>Click to posts</button>
    </>
  );
}

About.getLayout = function getLayout(page: ReactNode) {
  return <Layout>{page}</Layout>;
};

export default About;
