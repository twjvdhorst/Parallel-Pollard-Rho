using Cloo;
using System;
using System.Collections.Generic;
using System.IO;

namespace Parallel_Pollard_Rho.OpenCL
{
    // OpenCLProgram
    // Encapsulates the OpenCL context and queue and program. The constructor prepares
    // these, compiles the specified source file and attempts to initialize the GL interop
    // functionality.
    public class OpenCLProgram
    {
        private ComputeContext context;
        private ComputeCommandQueue queue;
        private ComputeProgram program;

        [System.Runtime.InteropServices.DllImport("opengl32", SetLastError = true)]
        private static extern IntPtr wglGetCurrentDC();

        public ComputeContext Context { get { return context; } }
        public ComputeCommandQueue Queue { get { return queue; } }
        public ComputeProgram Program { get { return program; } }

        /// <summary>
        /// Creates an instance of OpenCLProgram, consisting of all kernel methods in all the specified files.
        /// </summary>
        public OpenCLProgram(params string[] filePaths)
        {
            if (filePaths.Length < 1)
            {
                throw new ArgumentException("Specify at least one source file to compile.");
            }

            // Pick first platform
            SelectBestDevice();
            // Load OpenCL code
            string clSource = "";
            foreach (string filePath in filePaths)
            {
                try
                {
                    var streamReader = new StreamReader(filePath);
                    clSource += streamReader.ReadToEnd();
                    streamReader.Close();
                }
                catch
                {
                    throw new Exception("File not found:\n" + filePath);
                }
            }

            // Create program with OpenCL code
            program = new ComputeProgram(context, clSource);
            // Compile OpenCL code
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch
            {
                throw new Exception("Error in kernel code:\n" + program.GetBuildLog(context.Devices[0]));
            }
            // create a command queue with first gpu found
            queue = new ComputeCommandQueue(context, context.Devices[0], 0);
        }

        public void SelectBestDevice()
        {
            // This function attempts to find the best platform / device for OpenCL code execution.
            // The best device is typically not the CPU, nor an integrated GPU. If no GPU is found,
            // the CPU will be used, but this may limit compatibility, especially for the interop
            // functionality, but sometimes also for floating point textures.
            int bestPlatform = -1, bestDevice = -1, bestScore = -1;
            for (int i = 0; i < ComputePlatform.Platforms.Count; i++)
            {
                var platform = ComputePlatform.Platforms[i];
                for (int j = 0; j < platform.Devices.Count; j++)
                {
                    var device = platform.Devices[j];
                    if (device.Type == ComputeDeviceTypes.Gpu)
                    {
                        // Found a GPU device, prefer this over integrated graphics
                        int score = 1;
                        if (!platform.Name.Contains("Intel")) score = 10;
                        if (score > bestScore)
                        {
                            bestPlatform = i;
                            bestDevice = j;
                            bestScore = score;
                        }
                    }
                    else if (bestPlatform == -1)
                    {
                        // Found an OpenCL device, but not a GPU, better than nothing
                        bestPlatform = i;
                        bestDevice = j;
                    }
                }
            }
            if (bestPlatform > -1)
            {
                var platform = ComputePlatform.Platforms[bestPlatform];
                Console.Write($"Initializing OpenCL... {platform.Name} ({platform.Profile}).\n");
                // Try to enable gl interop functionality
                try
                {
                    var ctx = (OpenTK.Graphics.IGraphicsContextInternal)OpenTK.Graphics.GraphicsContext.CurrentContext;
                    IntPtr glHandle = ctx.Context.Handle;
                    IntPtr wglHandle = wglGetCurrentDC();
                    var p1 = new ComputeContextProperty(ComputeContextPropertyName.Platform, platform.Handle.Value);
                    var p2 = new ComputeContextProperty(ComputeContextPropertyName.CL_GL_CONTEXT_KHR, glHandle);
                    var p3 = new ComputeContextProperty(ComputeContextPropertyName.CL_WGL_HDC_KHR, wglHandle);
                    List<ComputeContextProperty> props = new List<ComputeContextProperty>() { p1, p2, p3 };
                    ComputeContextPropertyList Properties = new ComputeContextPropertyList(props);
                    context = new ComputeContext(ComputeDeviceTypes.Gpu, Properties, null, IntPtr.Zero);
                }
                catch
                {
                    // if this failed, we'll do without gl interop
                    try
                    {
                        context = new ComputeContext(ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);
                    }
                    catch
                    {
                        // failed to initialize a valid context; report
                        throw new Exception("Failed to initialize OpenCL context");
                    }
                }
            }
            else
            {
                // failed to find an OpenCL device; report
                throw new Exception("Failed to initialize OpenCL");
            }
        }
    }
}
