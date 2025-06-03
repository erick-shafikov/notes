import Link from 'next/link';

export default () => {
  return (
    <ul>
      <li>
        <Link href="/">Home</Link>
      </li>
      <li>
        <Link href="/posts">Posts</Link>
      </li>

      <li>
        <Link href="/about">About</Link>
      </li>
    </ul>
  );
};
