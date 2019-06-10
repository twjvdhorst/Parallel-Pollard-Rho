using System;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    static class Extentions
    {
        public static BigInteger Next(this Random rng, BigInteger maxVal)
        {
            byte[] bytes = maxVal.ToByteArray();
            BigInteger result;
            
            rng.NextBytes(bytes);
            // Force sign bit to positive.
            bytes[bytes.Length - 1] &= (byte)0x7F;
            result = new BigInteger(bytes);

            return result % maxVal;
        }

        /// <summary>
        /// Converts a BigInteger to an array of uints, rather than an array of bytes.
        /// </summary>
        public static uint[] ToUintArray(this BigInteger input)
        {
            byte[] byteArray = input.ToByteArray();
            int iterationCount = byteArray.Length / 4;
            if (byteArray.Length % 4 != 0)
                iterationCount++;

            byteArray = byteArray.PadWithDefaultForLength<byte>(4 * iterationCount);
            uint[] result = new uint[iterationCount];
            for (int i = 0, j = 0; i < iterationCount; i++, j += 4)
            {
                uint currUint = 0;
                currUint |= byteArray[j + 3];
                currUint = (currUint << 8) | byteArray[j + 2];
                currUint = (currUint << 8) | byteArray[j + 1];
                currUint = (currUint << 8) | byteArray[j];

                result[i] = currUint;
            }

            return result;
        }

        /// <summary>
        /// Converts a uint array to a BigInteger.
        /// </summary>
        public static BigInteger ToBigInteger(this uint[] input)
        {
            byte[] byteArray = new byte[4 * input.Length];
            for (int i = 0, j = 0; i < input.Length; i++, j += 4)
            {
                byteArray[j] = (byte)input[i];
                byteArray[j + 1] = (byte)(input[i] >> 8);
                byteArray[j + 2] = (byte)(input[i] >> 16);
                byteArray[j + 3] = (byte)(input[i] >> 24);
            }
            
            return new BigInteger(byteArray);
        }

        /// <summary>
        /// Returns a T array of size newLength, with the same values as before, but with default(T) values appended to it.
        /// </summary>
        public static T[] PadWithDefaultForLength<T>(this T[] input, int newLength)
        {
            if (newLength <= input.Length)
                return input;

            T[] result = new T[newLength];
            
            for (int j = 0; j < input.Length; j++)
            {
                result[j] = input[j];
            }

            for (int j = input.Length; j < newLength; j++)
            {
                result[j] = default(T);
            }

            return result;
        }
    }
}
