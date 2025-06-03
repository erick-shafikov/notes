import { NextFetchEvent, NextResponse, NextRequest } from 'next/server';

export default function middleware(
  request: NextRequest,
  event: NextFetchEvent
) {
  // использование event - метод waitUntil
  /* event.waitUntil(
    fetch('https://my-analytics-platform.com', {
      method: 'POST',
      body: JSON.stringify({ pathname: request.nextUrl.pathname }),
    })
  ); 
  
  return NextResponse.next();
  */

  // Сообщение запроса
  return new NextResponse(JSON.stringify({ message: 'authentication failed' }));
}

// See "Matching Paths" below to learn more
export const config = {
  matcher: ['/about/middleware-test/'],
};
// второй вариант обработки путей

/* export function middleware(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith('/about')) {
    return NextResponse.rewrite(new URL('/about-2', request.url))
  }
  if (request.nextUrl.pathname.startsWith('/dashboard')) {
    return NextResponse.rewrite(new URL('/dashboard/user', request.url))
  }
} */

/* 
NextRequest{
  cookies : {
    get:() {},
    getAll() {},
    set(){},
    delete(){},
    clear(){}.
  },
  nextUrl {
    basePath,
    trailingSlash,
    buildId,
    url
  }
  ip,
  geo :{
    city, 
    country
  }
}
 */

/* 
NextResponse{
  cookies : {
    get:() {},
    getAll() {},
    set(){},
    delete(){},
    clear(){}.
  },
  redirect(),
  rewrite(),
  next(),
}
*/
