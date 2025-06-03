import { Dispatch, SetStateAction } from 'react';

export type TPost = {
  body: string;
  id: string;
  title: string;
  userId: string;
};

export const setPostsFetcher = async <T extends TPost>(
  cb: Dispatch<SetStateAction<T[]>>
) => {
  const res = await fetch('https://jsonplaceholder.typicode.com/posts');
  const posts = await res.json();

  cb(posts);
};

export const fetchPosts = async (): Promise<TPost[]> => {
  const res = await fetch('https://jsonplaceholder.typicode.com/posts');
  const posts = await res.json();

  return posts;
};

export const getSinglePost = async (id: string): Promise<TPost> => {
  // если запрос идет из getStaticProps, то нужно добавить process.env.API, обратиться к api по полной строке
  const res = await fetch(process.env.API + `/api/post/${id}`);
  const json = await res.json();

  return json;
};

export const getPost = async (id: string): Promise<TPost> => {
  const res = await fetch(`https://jsonplaceholder.typicode.com/posts/${id}`);
  const json = await res.json();

  return json;
};
