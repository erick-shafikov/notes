import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  try {
    res.status(200).json({ middleware: 'from middleware' });
  } catch (err) {
    console.log(err);
  }
}
