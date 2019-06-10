using Parallel_Pollard_Rho.OpenCL;
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;

namespace Parallel_Pollard_Rho
{
    class SpecialPointCollector
    {
        // specialPoints keeps track of all returned special points, together with all corresponding
        // starting points used to get to the special points.
        private IDictionary<BigInteger, List<BigInteger>> specialPoints = new Dictionary<BigInteger, List<BigInteger>>();

        private int wordsPerNumber;

        private OpenCLBuffer<uint> gpuSpecialPointsBuffer;
        private Pollard_Rho pRho;
        private StartingPointGenerator startingPointGenerator;

        public SpecialPointCollector(int wordsPerNumber, Pollard_Rho pRho, OpenCLBuffer<uint> gpuSpecialPointsBuffer, StartingPointGenerator startingPointGenerator)
        {
            this.wordsPerNumber = wordsPerNumber;
            this.pRho = pRho;
            this.gpuSpecialPointsBuffer = gpuSpecialPointsBuffer;
            this.startingPointGenerator = startingPointGenerator;
        }

        /// <summary>
        /// Collects (starting point, special point) "tuples" from the gpuSpecialPointsBuffer and puts
        /// them in the specialPoints dictionary.
        /// </summary>
        public void CollectSpecialPoints(int amount)
        {
            // Put the data gotten from the GPU in the specialPoints dictionary.
            gpuSpecialPointsBuffer.CopyFromDevice();
            // TODO: Start seperate thread for this.
            uint[] gpuBufferArray = gpuSpecialPointsBuffer.CPUBuffer;
            FillSpecialPointsDict(gpuBufferArray, amount);
        }

        /// <summary>
        /// Searches for special points which were found in 2 or more chains, and calculates the result
        /// to the DLP from them.
        /// </summary>
        public BigInteger? FindCollision()
        {
            int counter = 0;
            foreach (var sp in specialPoints.Keys)
            {
                if (specialPoints[sp].Count > 1)
                {
                    BigInteger startingPoint0 = specialPoints[sp][0];
                    BigInteger startingPoint1 = specialPoints[sp][1];
                    Tuple<BigInteger, BigInteger> rep1 = null, rep2 = null;
                    Parallel.Invoke(() =>
                    {
                        rep1 = pRho.GetRepresentation(sp, startingPoint0, startingPointGenerator.StartingPoints[startingPoint0]);
                    },
                    () =>
                    {
                        rep2 = pRho.GetRepresentation(sp, startingPoint1, startingPointGenerator.StartingPoints[startingPoint1]);
                    });

                    BigInteger? result = pRho.GetResult(rep1, rep2);
                    if (result == null)
                        // The given starting points cannot be used for calculating an answer, so delete
                        // one of them.
                        specialPoints[sp].Remove(startingPoint0);

                    return result;
                }

                counter++;
            }

            return null;
        }

        /// <summary>
        /// Given an array, containing (starting point, special point) "tuples", fills the specialPoints dictionary
        /// with entries indicating that the given special point was found via the corresponding starting point.
        /// </summary>
        /// <param name="bufferArray"></param>
        /// <param name="amount"></param>
        private void FillSpecialPointsDict(uint[] bufferArray, int amount)
        {
            uint[] point = new uint[wordsPerNumber];
            for (int i = 0; i < 2 * amount; i += 2)
            {
                Array.Copy(bufferArray, i * wordsPerNumber, point, 0, wordsPerNumber);
                BigInteger startingPoint = point.ToBigInteger();
                Array.Copy(bufferArray, (i + 1) * wordsPerNumber, point, 0, wordsPerNumber);
                BigInteger specialPoint = point.ToBigInteger();

                if (!specialPoints.ContainsKey(specialPoint))
                    specialPoints[specialPoint] = new List<BigInteger>();

                if (startingPoint != 0)
                    specialPoints[specialPoint].Add(startingPoint);
            }
        }
    }
}
