using Parallel_Pollard_Rho.OpenCL;
using System;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    class DLPSolver
    {
        public const int NUM_GPU_THREADS = 2048;

        private BigInteger modulus, modulusPrime, generator, order, element;
        private int rAsPower, wordsPerNumber;

        private OpenCLProgram program = new OpenCLProgram("../../OpenCL/Parallel-Pollard-Rho.cl");

        private OpenCLBuffer<uint> gpuStartingPointsBuffer;
        private OpenCLBuffer<uint> gpuSpecialPointsBuffer;
        private StartingPointGenerator startingPointGenerator;
        private SpecialPointCollector specialPointCollector;
        
        public DLPSolver(InputTuple input)
        {
            modulus = input.Modulus;

            if (modulus % 2 == 0)
            {
                throw new NotImplementedException("At the moment, it is not possible to use an even number for a modulus.");
            }

            generator = input.Generator;
            order = input.Order;
            element = input.Element;
            
            wordsPerNumber = modulus.ToUintArray().Length;
            rAsPower = 32 * wordsPerNumber;
            
            Pollard_Rho pRho = new Pollard_Rho(input, rAsPower);

            // Initialize the startingPointGenerator.
            gpuStartingPointsBuffer = new OpenCLBuffer<uint>(program, new uint[4 * NUM_GPU_THREADS * wordsPerNumber]);
            startingPointGenerator = new StartingPointGenerator(input, rAsPower, wordsPerNumber, pRho, program, gpuStartingPointsBuffer);

            // Initialize the specialPointCollector.
            gpuSpecialPointsBuffer = new OpenCLBuffer<uint>(program, new uint[2 * wordsPerNumber * 4 * NUM_GPU_THREADS]);
            specialPointCollector = new SpecialPointCollector(wordsPerNumber, pRho, gpuSpecialPointsBuffer, startingPointGenerator);
        }

        /// <summary>
        /// Tries to solve the current instance of the DLP.
        /// </summary>
        public BigInteger? Solve()
        {
            // Fill the startingPointPool with starting points.
            BigInteger? answer = startingPointGenerator.FillStartingPointPool(4 * NUM_GPU_THREADS);
            if (answer != null)
                return answer;

            modulusPrime = -Utility.ExtendedEuclidesGCD1(BigInteger.One << 32, modulus).Item2;
            if (modulusPrime < 0)
                modulusPrime += (BigInteger.One << 32);

            BigInteger generatorMontgomery = ToMontgomery(generator);
            BigInteger elementMontgomery = ToMontgomery(element);

            // Initialize the kernel and buffers.
            OpenCLBuffer<int> countersBuffer;
            OpenCLKernel kernel = InitKernel(elementMontgomery, generatorMontgomery, out countersBuffer, out answer);
            if (answer != null)
                return answer;

            // Let the GPU do its job.
            int counter = 0;
            while (true)
            {
                Console.WriteLine($"Starting kernel {counter}.");
                kernel.Execute(new long[] { NUM_GPU_THREADS }, new long[] { 32 });
                kernel.WaitTillQueueFinish();
                Console.WriteLine($"Kernel {counter} finished.");

                countersBuffer.CopyFromDevice();
                Console.WriteLine($"Found special points: {countersBuffer[0]}.");

                // The number of used starting points has exceeded the threshold.
                if (countersBuffer[0] > NUM_GPU_THREADS)
                {
                    Console.WriteLine("Generating new starting points.");

                    answer = startingPointGenerator.FillStartingPointsBuffer(countersBuffer[0]);
                    if (answer != null)
                        return answer;

                    Console.WriteLine("Retrieving special points.");

                    specialPointCollector.CollectSpecialPoints(countersBuffer[0]);
                    answer = specialPointCollector.FindCollision();
                    if (answer != null)
                        return answer;

                    // Reset the counter.
                    countersBuffer[0] = 0;
                    countersBuffer.CopyToDevice();
                }
                counter++;
            }
        }

        /// <summary>
        /// Returns the given input in montgomery form.
        /// </summary>
        private BigInteger ToMontgomery(BigInteger input)
        {
            return (input << rAsPower) % modulus;
        }

        private OpenCLKernel InitKernel(BigInteger elementMontgomery, BigInteger generatorMontgomery, out OpenCLBuffer<int> gpuCounterBuffer, out BigInteger? answer)
        {
            // Make all inputs GPU ready, by converting them to uint arrays.
            // Note: each number (e.g. special points, the modulus etc.) will be represented with
            // wordsPerNumber uints (wordsPerNumber * 32 bits).
            uint[] gpuModulus = modulus.ToUintArray().PadWithDefaultForLength(wordsPerNumber); // TODO: Remove padding?

            uint[] gpuModulusPrime = modulusPrime.ToUintArray().PadWithDefaultForLength(wordsPerNumber);
            uint[] gpuElement = elementMontgomery.ToUintArray().PadWithDefaultForLength(wordsPerNumber);
            uint[] gpuGenerator = generatorMontgomery.ToUintArray().PadWithDefaultForLength(wordsPerNumber);

            // Input buffers.
            OpenCLBuffer<uint> gpuModulusBuffer = new OpenCLBuffer<uint>(program, gpuModulus);
            OpenCLBuffer<uint> gpuGeneratorBuffer = new OpenCLBuffer<uint>(program, gpuGenerator);
            OpenCLBuffer<uint> gpuElementBuffer = new OpenCLBuffer<uint>(program, gpuElement);

            // Buffers for local memory. There is room for an additional 2 numbers, which will be used to store
            // the generator and element in local memory.
            OpenCLBuffer<uint> gpuNumbersBuffer = new OpenCLBuffer<uint>(program, new uint[wordsPerNumber * (2 + 32)]);

            // Counter buffer.
            gpuCounterBuffer = new OpenCLBuffer<int>(program, new int[1]);

            // Buffers for saving numbers between kernel executions.
            uint[] startingPointsArray = startingPointGenerator.GetVerticalStartingPointsArray(NUM_GPU_THREADS, out answer);
            OpenCLBuffer<uint> gpuSavedNumbersBuffer = new OpenCLBuffer<uint>(program, startingPointsArray);
            OpenCLBuffer<uint> gpuUsedStartingPointBuffer = new OpenCLBuffer<uint>(program, startingPointsArray);
            OpenCLBuffer<long> gpuIterationCounts = new OpenCLBuffer<long>(program, new long[NUM_GPU_THREADS]);

            // Fill the gpuStartingPointBuffer.
            answer = startingPointGenerator.FillStartingPointsBuffer(4 * NUM_GPU_THREADS);

            OpenCLKernel kernel = new OpenCLKernel(program, "generate_chain");

            // Set the kernelarguments.
            kernel.SetArgument(0, gpuStartingPointsBuffer);
            kernel.SetArgument(1, gpuSavedNumbersBuffer);
            kernel.SetArgument(2, gpuUsedStartingPointBuffer);
            kernel.SetLocalArgument(3, gpuNumbersBuffer);
            kernel.SetArgument(4, gpuModulusBuffer);
            kernel.SetArgument<uint>(5, gpuModulusPrime[0]);
            kernel.SetArgument(6, gpuGeneratorBuffer);
            kernel.SetArgument(7, gpuElementBuffer);

            kernel.SetArgument(8, gpuSpecialPointsBuffer);

            kernel.SetArgument<int>(9, wordsPerNumber);
            kernel.SetArgument(10, gpuCounterBuffer);

            kernel.SetArgument(11, gpuIterationCounts);
            kernel.SetArgument<long>(12, 1 << (Program.K + 4)); // Maximum chain length is 16 * 2^k.
            kernel.SetArgument<int>(13, Program.K / 32); // Value of k, in words.
            kernel.SetArgument<int>(14, Program.K % 32); // Remaining value of k.

            return kernel;
        }
    }
}
