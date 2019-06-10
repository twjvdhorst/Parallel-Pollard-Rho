// Checks wheter first is greater than or equal to second.
bool geq(uint* first, __global uint* second, int words_per_number)
{
    for (int i = words_per_number - 1; i >= 0; i--)
    {
        if (first[i] > second[i])
        {
            return true;
        }
        else if (first[i] < second[i])
        {
            return false;
        }
    }

	return true;
}

// Subtracts second from first, in-place to prevent copying. Note: the result will
// be a number with words_per_number words, meaning any excess carry will be lost.
// This is not a problem for mont_mul, as resulting numbers should never exceed
// words_per_number in size.
void subtract(uint* first, __global uint* second, int words_per_number)
{
    int carry_pre = 0;
    int carry_post = 0;
    for (int i = 0; i < words_per_number; i++)
    {
        // First calculate the carry resulting from the current loop. This has to
        // be done before the subtraction, as first is updated in-place during the
        // subtraction, causing carry calculation to be off otherwise.
        carry_post = ((long)first[i] + (long)carry_pre < (long)second[i]) ? -1 : 0;

        // Perform the subtraction. Note that underflow is not a problem here, as
        // the numbers will wrap around.
        first[i] = (uint)(first[i] - second[i] + carry_pre);

        // Update the carry for the next subtraction.
        carry_pre = carry_post;
    }
}

// A CIOS approuch for the Montgomery multiplication algorithm.
// For reference, see https://iacr.org/archive/ches2005/006.pdf.
// m_prime_0 is passed as a ulong, but cannot be larger than UINT_MAX. It is treated
// as a ulong to save some typecasting.
// Unfortunately, mont_mul cannot be done in-place, meaning the result array has to be
// copied back to first at the end. This also means that a result array with a compile-time
// constant size has to be created. As for this use-case numbers have a maximum of 2048
// bits, the result array will have size 64 + 2 = 66.
void mont_mul(__local uint* first, __local uint* second, __global uint* modulus, 
ulong m_prime_0, int words_per_number, int local_width)
{
    // During computation, a maximum of 66 array cells are needed, but at the end, 
    // only the first 64 cells can be non-zero (as result will be < modulus).
    uint* result[66] = {0};
    for (int i = 0; i < words_per_number; i++)
    {
        ulong intermediate;
        // u and v will never be greater than UINT_MAX, but keeping them as a ulong
        // saves some typecasting.
        ulong u = 0;
        ulong v;
        for (int j = 0; j < words_per_number; j++)
        {
            intermediate = (ulong)first[j * local_width] * (ulong)second[i * local_width] + (ulong)result[j] + u;
            u = intermediate >> 32;
            v = intermediate & UINT_MAX;
            result[j] = (uint)v;
        }

        intermediate = (ulong)result[words_per_number] + u;
        u = intermediate >> 32;
        v = intermediate & UINT_MAX;
        result[words_per_number] = (uint)v;
        result[words_per_number + 1] = (uint)u;

        // q will never be greater than UINT_MAX, but is kept as a ulong to save some
        // typecasting.
        ulong q = ((ulong)result[0] * m_prime_0) & UINT_MAX;
        intermediate = (ulong)result[0] + (ulong)modulus[0] * q;
        u = intermediate >> 32;
        v = intermediate & UINT_MAX;

        for (int j = 1; j < words_per_number; j++)
        {
            intermediate = (ulong)modulus[j] * q + (ulong)result[j] + u;
            u = intermediate >> 32;
            v = intermediate & UINT_MAX;
            result[j - 1] = (uint)v;
        }

        intermediate = (ulong)result[words_per_number] + u;
        u = intermediate >> 32;
        v = intermediate & UINT_MAX;
        result[words_per_number - 1] = (uint)v;
        result[words_per_number] = (uint)((ulong)result[words_per_number + 1] + u);
    }

    // Here, result < 2 * modulus. If result >= modulus, subtract the modulus, 
    // which will put result to be less than modulus.
    if (result[words_per_number] > 0 || geq(result, modulus, words_per_number))
    {
        subtract(result, modulus, words_per_number);
    }

    // Copy the result into first.
    for (int i = 0; i < words_per_number; i++)
    {
        first[i * local_width] = result[i];
    }
}

// Puts the used_starting_point, special_point combo in the special_points buffer and returns
// a pointer to a new starting point that can be used.
__global uint* handle_special_point(__local uint* special_point, __global uint* special_points,
__global uint* used_starting_point, __global uint* starting_points_buffer, __global int* counter, 
int words_per_number)
{
    int global_width = get_global_size(0);
    int local_width = get_local_size(0) + 2;
    int index = atomic_inc(&counter[0]);
    int special_point_index = 2 * index * words_per_number;
    __global uint* save_location = &special_points[special_point_index];

    // Save the used_starting_point, special_point combo in the special_points buffer.
    for (int i = 0; i < words_per_number; i++)
    {
        save_location[i] = used_starting_point[i * global_width];
        save_location[i + words_per_number] = special_point[i * local_width];
    }

    int starting_point_index = index * words_per_number;
    return &starting_points_buffer[starting_point_index];
}

void copy_new_starting_point(__global uint* new_starting_point, __global uint* used_starting_point_buffer,
__local uint* number, int words_per_number)
{
    int global_id = get_global_id(0);
    int global_width = get_global_size(0);
    int local_width = get_local_size(0) + 2;
    __global uint* save_location = &used_starting_point_buffer[global_id];

    // Save the new_starting_point in the used_starting_point_buffer and
    // in the local number buffer.
    for (int i = 0; i < words_per_number; i++)
    {
        save_location[i * global_width] = new_starting_point[i];
        number[i * local_width] = new_starting_point[i];
    }
}

// Checks if a point is special, by checking if the k trailing bits are zero.
// Here, k is split in k_cells (k / 32) and k_bits (k % 32).
bool is_special(__local uint* number, int k_cells, int k_bits, int local_width)
{
	// First check if the first k_cells words of the number are zero.
	for (int i = 0; i < k_cells; i++)
	{
		if (number[i * local_width] != 0)
			return false;
	}

	// Check if the k_bits bits after the first k_cells words are zero.
	return (number[k_cells * local_width] & ((1 << k_bits) - 1)) == 0;
}

// Iteration kernel. It gets the saved number from saved_numbers_buffer, and continues iterating
// until a special point is found. If a special point is found, it is send back to the CPU, together
// with the used starting point (stored in used_starting_point_buffer). After this, a new starting
// point is picked from starting_points, after which iteration can continue.
__kernel void generate_chain(__global uint* starting_points, __global uint* saved_numbers_buffer, 
__global uint* used_starting_point_buffer, __local uint* numbers_buffer, __global uint* modulus, 
uint modulus_prime_0, __global uint* generator, __global uint* element, __global uint* special_points, 
int words_per_number, __global int* counter, __global long* current_iterations, long max_iterations,
int k_cells, int k_bits)
{
    // Get the saved number that this thread will start iterating with.
    int global_id = get_global_id(0);
    __global uint* saved_number = &saved_numbers_buffer[global_id];

    // Get the address of the first cell in local memory meant for this thread.
    // Note: the first two addresses will be for the generator and element. These
    // will be set by the thread with local_id 0.
    int local_id = get_local_id(0);
    __local uint* number = &numbers_buffer[local_id + 2];

    // Put the saved number into local memory.
    int global_width = get_global_size(0);
    int local_width = get_local_size(0) + 2;
    for (int i = 0; i < words_per_number; i++)
    {
        number[i * local_width] = saved_number[i * global_width];
    }

    // Put the generator and element in local memory. local_width here is the number
    // of numbers that fit in local memory, including the generator and element.
    if (local_id == 0)
    {
        for (int i = 0; i < words_per_number; i++)
        {
            numbers_buffer[i * local_width] = generator[i];
            numbers_buffer[i * local_width + 1] = element[i];
        }
    }

    // Make sure all relevent numbers are in local memory before continuing.
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Generate the chain.
    long current_iteration = current_iterations[global_id]; // The current iteration of the chain, persistent between kernel executions.
    int iteration_count = 0;                                // The current iteration of the chain, in this kernel execution.
    int partition;
    __local uint* multiplier;
    while (iteration_count < 50000)
    {
        // TODO: Take more words into account for partition. Perhaps the whole number 
        // with MontMul, with modulus 3?
        partition = (int)(number[0] % 3);

        if (partition == 0)
        {
            // multiplier will be the generator.
            multiplier = &numbers_buffer[0];
            
        }
        else if (partition == 1)
        {
            // multiplier will be the element.
            multiplier = &numbers_buffer[1];
        }
        else
        {
            // multiplier will be the number itself.
            multiplier = &number[0];
        }

        // Calculate new point.
        mont_mul(number, multiplier, modulus, modulus_prime_0, words_per_number, local_width);

        current_iteration++;
        iteration_count++;
        
        // Special point condition. Checks if the trailing k bits are 0 (k is split up in
		// k_cells (k / 32) and k_bits (k % 32)). Also treat the point as special if the 
		// max iterations for a chain has been reached.
        if (is_special(number, k_cells, k_bits, local_width) || current_iteration > max_iterations)
        {
            // Store special point data and get new starting point.
            __global uint* new_starting_point = handle_special_point(number, special_points, &used_starting_point_buffer[global_id], starting_points, counter, words_per_number);
            // Save new starting point in the right buffers.
            copy_new_starting_point(new_starting_point, used_starting_point_buffer, number, words_per_number);
            current_iteration = 0;
        }
    }

    // Save the current point in global memory, so it can be used
    // again in the next kernel call.
    for (int i = 0; i < words_per_number; i++)
    {
        saved_number[i * global_width] = number[i * local_width];
    }
    current_iterations[global_id] = current_iteration;
}

// Kernel that copies new starting points into the pool of starting points used by the
// generate_chain kernel. starting_points is a pointer to this pool, new_starting_points
// is a pointer to the new starting points, start_index is the first index in the 
// starting_points array where the new starting points have to be placed, count is the
// amount of new starting points and words_per_number is the amount of words (uints) each
// starting point takes up.
__kernel void add_new_starting_points(__global uint* starting_points, 
__global uint* new_starting_points, int count, int words_per_number)
{
    // Divide the work over the available threads.
    int thread_count = get_global_size(0);
    int thread_id = get_global_id(0);
    int copy_count = count / thread_count;
    int copy_start_index = thread_id * copy_count * words_per_number;
    for (int i = 0; i < copy_count; i++)
    {
        for (int j = 0; j < words_per_number; j++)
        {
            starting_points[i * words_per_number + copy_start_index + j] 
                = new_starting_points[i * words_per_number + copy_start_index + j];
        }
    }
    
    // Copy the last cells of new_starting_points (the ones left over because of the 
    // division for copy_count).
    int leftover_count = count % thread_count;
    if (thread_id < leftover_count)
    {
        copy_start_index = (count - thread_id - 1) * words_per_number;
        for (int i = 0; i < words_per_number; i++)
        {
            starting_points[copy_start_index + i]
                = new_starting_points[copy_start_index + i];
        }
    }
}