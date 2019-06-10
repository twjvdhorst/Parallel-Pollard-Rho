using Cloo;

namespace Parallel_Pollard_Rho.OpenCL
{
    // OpenCL kernel
    // Encapsulates an OpenCL kernel. Multiple kernels may exist in the same program.
    // SetArgument methods are provided to conveniently set various argument types.
    public class OpenCLKernel
    {
        private ComputeKernel kernel;
        private ComputeCommandQueue queue;

        public OpenCLKernel(OpenCLProgram ocl, string kernelName)
        {
            // make a copy of the queue descriptor
            queue = ocl.Queue;
            // load chosen kernel from program
            kernel = ocl.Program.CreateKernel(kernelName);
        }

        public void SetArgument<T>(int i, T v) where T : struct { kernel.SetValueArgument(i, v); }
        public void SetArgument(int i, OpenCLBuffer<int> v) { kernel.SetMemoryArgument(i, v.GPUBuffer); }
        public void SetArgument(int i, OpenCLBuffer<uint> v) { kernel.SetMemoryArgument(i, v.GPUBuffer); }
        public void SetArgument(int i, OpenCLBuffer<long> v) { kernel.SetMemoryArgument(i, v.GPUBuffer); }

        public void SetLocalArgument(int i, OpenCLBuffer<uint> v) { kernel.SetLocalArgument(i, v.GPUBuffer.Size); }
        
        public void Execute(long[] workSize, long[] localSize = null)
        {
            queue.Execute(kernel, null, workSize, localSize, null);
        }

        public void WaitTillQueueFinish()
        {
            queue.Finish();
        }
    }
}
