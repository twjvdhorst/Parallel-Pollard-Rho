using System;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    static class Utility
    {
        /// <summary>
        /// Given two integers a and b, with gcd(a, b) = 1, finds integers x and y such that ax + by = 1.
        /// </summary>
        public static Tuple<BigInteger, BigInteger> ExtendedEuclidesGCD1(BigInteger x, BigInteger y)
        {
            if (y == 0)
                return new Tuple<BigInteger, BigInteger>(x, 1);

            var abPrime = ExtendedEuclidesGCD1(y, x % y);
            BigInteger a = abPrime.Item2;
            BigInteger b = abPrime.Item1 - x / y * abPrime.Item2;
            return new Tuple<BigInteger, BigInteger>(a, b);
        }
    }
}
