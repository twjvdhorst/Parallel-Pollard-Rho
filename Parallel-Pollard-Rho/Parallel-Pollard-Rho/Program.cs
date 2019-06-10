using System;
using System.Numerics;

namespace Parallel_Pollard_Rho
{
    class Program
    {
        private static int k;
        public static int K { get { return k; } }

        public static Random Random = new Random();

        [STAThread]
        static void Main(string[] args)
        {
            new Program();
        }

        public Program()
        {
            InputTuple DLP = GetDLP();

            while (true)
            {
                Console.WriteLine("What value k for the special point condition would you like to use?");
                bool parseSuccess = int.TryParse(Console.ReadLine(), out k);
                if (parseSuccess)
                {
                    if (k >= 0)
                        break;

                    Console.WriteLine("Please enter a non-negative number.");
                    continue;
                }
                
                Console.WriteLine("The input could not be parsed. Please try again.");
            }
            
            DLPSolver solver = new DLPSolver(DLP);
            BigInteger? answer = solver.Solve();

            Console.WriteLine($"{DLP}.\nSolution: {answer}.\n");
            Console.ReadLine();
        }

        private InputTuple GetDLP()
        {
            string[] inputNames = new string[]
            {
                "modulus (p)",
                "generator (g)",
                "order of the generator (q)",
                "element (y)"
            };

            BigInteger[] inputs = new BigInteger[4];
            for (int i = 0; i < 4; i++)
            {
                while (true)
                {
                    Console.WriteLine($"Please give the {inputNames[i]}.");
                    bool parseSuccess = BigInteger.TryParse(Console.ReadLine(), out inputs[i]);
                    if (parseSuccess)
                    {
                        if (inputs[i] > 0)
                            break;

                        Console.WriteLine("Please give a positive number.");
                        continue;
                    }

                    Console.WriteLine("The input could not be parsed. Try again.");
                }
            }

            return new InputTuple(inputs[0], inputs[1], inputs[2], inputs[3]);
        }
    }

    struct InputTuple
    {
        private BigInteger modulus, generator, order, element;

        public BigInteger Modulus { get { return modulus; } }
        public BigInteger Generator { get { return generator; } }
        public BigInteger Order { get { return order; } }
        public BigInteger Element { get { return element; } }

        public InputTuple(BigInteger modulus, BigInteger generator, BigInteger order, BigInteger element)
        {
            this.modulus = modulus;
            this.generator = generator;
            this.order = order;
            this.element = element;
        }

        public override string ToString()
        {
            return $"Modulus: {modulus}, Generator: {generator}, Order: {order}, Element: {element}";
        }
    }
}
