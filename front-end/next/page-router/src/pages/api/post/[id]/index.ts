import { getPost } from '@/services/api';
import type { NextApiRequest, NextApiResponse } from 'next';

type Data = {
  body: string;
  id: string;
  title: string;
  userId: string;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<Data>
) {
  const { id } = req.query;
  try {
    const data = await getPost(id as string);
    if (data) {
      res.status(200).json(data as Data);
    }
  } catch (err) {
    console.log(err);
  }
}

/* export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
  },
  responseLimit: false,
  // Specifies the maximum allowed duration for this function to execute (in seconds)
  maxDuration: 5,
} */
