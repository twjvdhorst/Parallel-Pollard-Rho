using Cloo;

namespace Parallel_Pollard_Rho.OpenCL
{
    // OpenCLBuffer
    // Encapsulates an OpenCL ComputeBuffer and a host-side buffer. The data may exist on
    // the host and/or the device. Copying data between host and device is implemented in
    // methods CopyToDevice and CopyFromDevice. Access of the host-side data is supported
    // using the [] operator.
    public class OpenCLBuffer<T> where T : struct
    {
        // CPU/GPU buffer - wrapper around ComputerBuffer<T> and T[], based on Duncan Ogilvie
        private ComputeCommandQueue queue;
        private T[] cpuBuffer;
        private ComputeBuffer<T> gpuBuffer;

        public const int ON_DEVICE = 1;
        public const int ON_HOST = 2;
        public const int WRITE_ONLY = 4;
        public const int READ_ONLY = 8;
        public const int READ_WRITE = 16;

        public T[] CPUBuffer { get { return cpuBuffer; } }
        public ComputeBuffer<T> GPUBuffer { get { return gpuBuffer; } }

        public OpenCLBuffer(OpenCLProgram ocl, T[] buffer, int flags = ON_DEVICE + ON_HOST + READ_WRITE)
        {
            queue = ocl.Queue;
            int clflags = (int)ComputeMemoryFlags.UseHostPointer;
            if ((flags & READ_ONLY) > 0) clflags += (int)ComputeMemoryFlags.ReadOnly;
            if ((flags & WRITE_ONLY) > 0) clflags += (int)ComputeMemoryFlags.WriteOnly;
            if ((flags & READ_WRITE) > 0) clflags += (int)ComputeMemoryFlags.ReadWrite;
            cpuBuffer = buffer;
            if ((flags & ON_DEVICE) > 0)
            {
                gpuBuffer = new ComputeBuffer<T>(ocl.Context, (ComputeMemoryFlags)clflags, cpuBuffer);
                CopyToDevice();
            }
        }

        public void CopyToDevice()
        {
            queue.WriteToBuffer(cpuBuffer, gpuBuffer, true, null);
        }

        public void CopyFromDevice()
        {
            queue.ReadFromBuffer(gpuBuffer, ref cpuBuffer, true, null);
        }

        public T this[int index]
        {
            get { return cpuBuffer[index]; }
            set { cpuBuffer[index] = value; }
        }
    }
}
