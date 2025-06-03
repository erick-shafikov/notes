import Link from 'next/link';
import { ReactNode } from 'react';

export default function LayoutPost({ children }: { children: ReactNode }) {
  return (
    <>
      post layout
      <Link href="/post">To posts</Link>
      {children}
    </>
  );
}
