using Parallel_Pollard_Rho.OpenCL;
using System;
using System.Collections.Generic;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    class StartingPointGenerator
    {
        // startingPoints keeps track of all generated starting points and their representations.
        // Note that there will always be one representation. If more are found, they are used
        // inmediately to find an answer to the DLP, see CreateStartingPoints method.
        private IDictionary<BigInteger, Tuple<BigInteger, BigInteger>> startingPoints = new Dictionary<BigInteger, Tuple<BigInteger, BigInteger>>();

        private BigInteger modulus, generator, order, element;
        private int rAsPower, wordsPerNumber;
        private Pollard_Rho pRho;
        
        private OpenCLKernel kernel;
        private OpenCLBuffer<uint> newStartingPointsBuffer;

        private uint[] startingPointPool;

        public IDictionary<BigInteger, Tuple<BigInteger, BigInteger>> StartingPoints
        {
            get { return startingPoints; }
        }

        public StartingPointGenerator(InputTuple input, int rAsPower, int wordsPerNumber, Pollard_Rho pRho, OpenCLProgram program, OpenCLBuffer<uint> startingPointsBuffer)
        {
            this.modulus = input.Modulus;
            this.generator = input.Generator;
            this.order = input.Order;
            this.element = input.Element;
            this.rAsPower = rAsPower;
            this.wordsPerNumber = wordsPerNumber;
            this.pRho = pRho;

            this.kernel = new OpenCLKernel(program, "add_new_starting_points");
            this.newStartingPointsBuffer = new OpenCLBuffer<uint>(program, new uint[4 * DLPSolver.NUM_GPU_THREADS * wordsPerNumber]);

            this.startingPointPool = new uint[4 * DLPSolver.NUM_GPU_THREADS * wordsPerNumber];

            kernel.SetArgument(0, startingPointsBuffer);
            kernel.SetArgument(1, newStartingPointsBuffer);
        }
        
        /// <summary>
        /// Generates an array of starting points with the starting points laid out "vertically",
        /// when the array is seen as a 2d array.
        /// </summary>
        public uint[] GetVerticalStartingPointsArray(int amount, out BigInteger? answer)
        {
            // Generate new starting points, returning the answer to the DLP if one is found.
            ICollection<BigInteger> newStartingPoints = CreateStartingPoints(amount, out answer);
            uint[] startingPointsArray = CreateStartingPointsArray(newStartingPoints);
            uint[] resultArray = new uint[startingPointsArray.Length];

            // Put the starting points "vertically" in the resultArray.
            for (int i = 0; i < amount; i++)
            {
                for (int j = 0; j < wordsPerNumber; j++)
                {
                    resultArray[i + j * amount] = startingPointsArray[i * wordsPerNumber + j];
                }
            }

            return resultArray;
        }

        /// <summary>
        /// Generates new starting points and puts them in the given buffer on both the CPU and GPU.
        /// </summary>
        public BigInteger? FillStartingPointsBuffer(int amount)
        {
            // Put some starting points from the pool in the newStartingPointsBuffer buffer.
            Array.Copy(startingPointPool, newStartingPointsBuffer.CPUBuffer, amount * wordsPerNumber);
            newStartingPointsBuffer.CopyToDevice();
            
            kernel.SetArgument<int>(2, amount);
            kernel.SetArgument<int>(3, wordsPerNumber);
            kernel.Execute(new long[] { 32 });

            // TODO: Make async.
            BigInteger? answer;
            answer = FillStartingPointPool(amount);
            
            kernel.WaitTillQueueFinish();

            return answer;
        }

        /// <summary>
        /// Fills up the startingPointPool with starting points, where the first starting point is
        /// put at index 0.
        /// </summary>
        public BigInteger? FillStartingPointPool(int amount)
        {
            BigInteger? answer;

            // Generate new starting points, returning the answer to the DLP if one is found.
            ICollection<BigInteger> newStartingPoints = CreateStartingPoints(amount, out answer);
            if (answer != null)
                return answer;

            // Put the new starting points in the pool.
            uint[] newStartingPointsArray = CreateStartingPointsArray(newStartingPoints);
            Array.Copy(newStartingPointsArray, startingPointPool, newStartingPointsArray.Length);

            return null;
        }

        /// <summary>
        /// Fills up the startingPoints dictionary with random starting points and their representations.
        /// Returns the newly generated startingpoints, or null if an answer has been found, in which case
        /// the out parameter answer has the value.
        /// </summary>
        private ICollection<BigInteger> CreateStartingPoints(int amount, out BigInteger? answer)
        {
            answer = null;
            ICollection<BigInteger> startingPoints = new List<BigInteger>();

            for (int i = 0; i < amount; i++)
            {
                BigInteger a = Program.Random.Next(order);
                BigInteger b = Program.Random.Next(order);
                BigInteger z = BigInteger.ModPow(generator, a, modulus) * BigInteger.ModPow(element, b, modulus);

                z = ToMontgomery(z);
                try
                {
                    this.startingPoints.Add(z, new Tuple<BigInteger, BigInteger>(a, b));
                    startingPoints.Add(z);
                }
                catch (ArgumentException)
                {
                    // A startingpoint with the same value as z has already been added.
                    Tuple<BigInteger, BigInteger> representation = this.startingPoints[z];
                    if (representation.Item2 == b)
                    {
                        // The startingpoint has the same b in the represenation as z, 
                        // meaning (d - b) == 0 (with d being the b of startingpoint).
                        // Therefore, the collision won't lead to a result, so just
                        // generate another startingpoint to replace this generated one.
                        i--;
                        continue;
                    }

                    // The startingpoint has a different representation as z, yet has the
                    // same value. Use this to calculate the result to the DLP.
                    answer = pRho.GetResult(representation, new Tuple<BigInteger, BigInteger>(a, b));
                    return null;
                }
            }

            return startingPoints;
        }

        /// <summary>
        /// Converts a given collection of starting points into a uint array.
        /// </summary>
        private uint[] CreateStartingPointsArray(ICollection<BigInteger> startingPoints)
        {
            uint[] result = new uint[wordsPerNumber * startingPoints.Count];
            int i = 0;
            foreach (BigInteger startingPoint in startingPoints)
            {
                uint[] uintArray = startingPoint.ToUintArray().PadWithDefaultForLength<uint>(wordsPerNumber);

                for (int j = 0; j < uintArray.Length; j++)
                {
                    result[i + j] = uintArray[j];
                }

                i += wordsPerNumber;
            }

            return result;
        }

        /// <summary>
        /// Returns the given input in montgomery form.
        /// </summary>
        private BigInteger ToMontgomery(BigInteger input)
        {
            return (input << rAsPower) % modulus;
        }
    }
}
