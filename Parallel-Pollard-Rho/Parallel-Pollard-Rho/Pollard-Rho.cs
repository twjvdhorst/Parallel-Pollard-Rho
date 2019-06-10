using System;
using System.Collections.Generic;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    class Pollard_Rho
    {
        private BigInteger modulus, generator, order, element, r, modulus_prime;
        private int rAsPower;

        public Pollard_Rho(InputTuple input, int rAsPower)
        {
            this.modulus = input.Modulus;
            this.generator = input.Generator;
            this.order = input.Order;
            this.element = input.Element;

            this.rAsPower = rAsPower;
            this.r = BigInteger.One << rAsPower;
            this.modulus_prime = -Utility.ExtendedEuclidesGCD1(r, modulus).Item2;
        }

        /// <summary>
        /// Returns the result of a DLP, given two represenations of the same number.
        /// </summary>
        public BigInteger? GetResult(Tuple<BigInteger, BigInteger> rep1, Tuple<BigInteger, BigInteger> rep2)
        {
            BigInteger a1 = rep1.Item1, a2 = rep2.Item1;
            BigInteger b1 = rep1.Item2, b2 = rep2.Item2;

            // As the order q is prime, b2 - b1 is a unit iff b1 != b2.
            if (b1 == b2)
                return null;

            // The inverse of (b2 - b1) can be calculated as (b2 - b1)^{order - 2}, as the order q of the
            // generator is prime, meaning (b2 - b1) has order (order - 1).
            BigInteger result = (a1 - a2) * BigInteger.ModPow(b2 - b1, order - 2, order);
            result %= order;
            if (result < 0)
                result += order;

            return result;
        }

        /// <summary>
        /// Returns the representation of the given specialPoint, by constructing a chain from point to specialPoint.
        /// </summary>
        public Tuple<BigInteger, BigInteger> GetRepresentation(BigInteger specialPoint, BigInteger point, Tuple<BigInteger, BigInteger> representation)
        {
            var points = new List<BigInteger>();
            var resultRepresentation = new Tuple<BigInteger, BigInteger>(representation.Item1, representation.Item2);
            while (point != specialPoint)
            {
                var newPointInfo = F(point, resultRepresentation);
                point = newPointInfo.Item1;
                resultRepresentation = newPointInfo.Item2;
            }
            return resultRepresentation;
        }

        /// <summary>
        /// The iteration function.
        /// </summary>
        private Tuple<BigInteger, Tuple<BigInteger, BigInteger>> F(BigInteger point, Tuple<BigInteger, BigInteger> representation)
        {
            int partition = (int)((point & uint.MaxValue) % 3);
            Tuple<BigInteger, BigInteger> resultRepresentation;
            switch (partition)
            {
                case 0:
                    point = (point * generator) % modulus;
                    resultRepresentation = new Tuple<BigInteger, BigInteger>((representation.Item1 + 1) % order, representation.Item2);
                    break;
                case 1:
                    point = (point * element) % modulus;
                    resultRepresentation = new Tuple<BigInteger, BigInteger>(representation.Item1, (representation.Item2 + 1) % order);
                    break;
                case 2:
                    point = MontReduc(point * point);
                    resultRepresentation = new Tuple<BigInteger, BigInteger>((2 * representation.Item1) % order, (2 * representation.Item2) % order);
                    break;
                default:
                    throw new Exception("Something went wrong with calculating the partition.");
            }

            return new Tuple<BigInteger, Tuple<BigInteger, BigInteger>>(point, resultRepresentation);
        }
        
        /// <summary>
        /// Performs montgomery reduction.
        /// </summary>
        private BigInteger MontReduc(BigInteger input)
        {
            BigInteger m = ((input & (r - 1)) * modulus_prime) & (r - 1);
            BigInteger t = (input + m * modulus) >> rAsPower;
            if (t >= modulus)
                t -= modulus;
            return t;
        }
    }
}
