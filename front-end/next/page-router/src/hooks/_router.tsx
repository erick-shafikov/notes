// Router Api
import { useRouter } from 'next/router';

const Router = () => {
  const router = useRouter();

  //----------------------------------поля----------------------------------
  /*
  console.log(router.pathname); //string путь роута
  console.log(router.query); //object{} параметры строки в том числе и динамические
  console.log(router.asPath); //string как строка запроса представлена в браузере
  console.log(router.isFallback); //boolean если в ssp fallback === true
  console.log(router.basePath); //string
  console.log(router.locale); //string
  console.log(router.locales); //string[]
  console.log(router.defaultLocale); //string
  console.log(router.domainLocales); //Array<{domain, defaultLocale, locales}>
  console.log(router.isReady); //готова ли строка при использовании useEffect
  console.log(router.isPreview); //boolean находится ли страница в preview mode
 */

  //----------------------------------методы----------------------------------
  router.push('/', '/home', { scroll: true, shallow: false, locale: 'en' });
  //'/' - место перенаправления

  // можно использовать с объектом
  // router.push({
  // pathname: '/post/[pid]',
  // query: { pid: post.id },
  // })

  //'/home' - как отобразится в строке URL
  // { scroll: true, shallow: false, locale: 'en' } - доп параметры
  // shallow - обновляет путь без запуска getStaticProps getServerSideProps

  router.push('/'); //без добавления в history api браузера
  router.prefetch('/'); //не работает в dev режиме, обеспечивает более быстрый переход со страницы на страницу
  router.beforePopState(({ url, as, options }) => {
    //метод позволяет управлять логикой перенаправления, при true - происходит, при false - нет
    //принимает 3 объект из аргументов { url, as, options }
    return true;
  });
  router.back(); //возвращает назад window.history.back().
  router.reload(); //перезагружает страницу
};

/* !!! Переход по роутеру не сбрасывает состояние useSate()
что бы исправить это поведение. нужно воспользоваться параметром key в корневом компоненте
/_app
import { useRouter } from 'next/router'
export default function MyApp({ Component, pageProps }) {
  const router = useRouter()
  return <Component key={router.asPath} {...pageProps} />
} */

//----------------------------------router.events----------------------------------
// routeChangeStart(url, { shallow }) - событие, которое срабатывает при начале изменения
// routeChangeComplete(url, { shallow }) - событие, которое срабатывает при окончании изменения
// routeChangeError(err, url, { shallow }) - если произошла ошибка
// err.cancelled - при отмененном переходе
// beforeHistoryChange(url, { shallow }) - перед начало изменения истории
// hashChangeStart(url, { shallow }) - начал изменяться строка, но не страница
// hashChangeComplete(url, { shallow }) - изменилась строка, но не страница

//обработка на уровне корня приложения
/* pages/_app.js

import { useEffect } from 'react'
import { useRouter } from 'next/router'
export default function MyApp({ Component, pageProps }) {
  const router = useRouter()

  useEffect(() => {
    const handleRouteChange = (url, { shallow }) => {
      console.log(
        `App is changing to ${url} ${
          shallow ? 'with' : 'without'
        } shallow routing`
      )
    }

    router.events.on('routeChangeStart', handleRouteChange)
    // If the component is unmounted, unsubscribe
    // from the event with the `off` method:
    return () => {
      router.events.off('routeChangeStart', handleRouteChange)
    }
  }, [router])

  return <Component {...pageProps} />
}
*/

/* 

import { useEffect } from 'react'
import { useRouter } from 'next/router'

export default function MyApp({ Component, pageProps }) {
  const router = useRouter()

  useEffect(() => {
    const handleRouteChangeError = (err, url) => {
      if (err.cancelled) {
        console.log(`Route to ${url} was cancelled!`)
      }
    }

    router.events.on('routeChangeError', handleRouteChangeError)

    // If the component is unmounted, unsubscribe
    // from the event with the `off` method:
    return () => {
      router.events.off('routeChangeError', handleRouteChangeError)
    }
  }, [router])

  return <Component {...pageProps} />
}
*/
