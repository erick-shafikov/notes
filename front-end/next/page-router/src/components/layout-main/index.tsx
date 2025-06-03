// компонент главной обертки
import Navbar from '@/components/navbar';
import { ReactNode } from 'react';

export default function Layout({ children }: { children: ReactNode }) {
  //Здесь можно взаимодействовать с сервером, нельзя использовать getStaticProps or getServerSideProps
  return (
    <>
      <Navbar />
      {children}
    </>
  );
}
