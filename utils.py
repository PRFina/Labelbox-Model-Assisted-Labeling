import numpy as np
from dataclasses import dataclass

def random_block_assignment(width, height, value, block_size, rgb=False):
    """
    Generates a numpy array and randomly assigns blocks of elements to 3.

    Args:
        n: The number of rows in the array.
        m: The number of columns in the array.
        num_iterations: The number of iterations.
        block_size: The size of the square block to assign.

    Returns:
        A list of numpy arrays, one for each iteration.
    """
  
    shape = (height, width, 3) if rgb else (height, width)
    if rgb and not isinstance(value, list) and not len(value) == 3:
        raise ValueError("value needs to be a 3 element list if rgb=True")
    
    array = np.zeros(shape, dtype=np.uint8)
    start_row = np.random.randint(0, height - block_size)
    start_col = np.random.randint(0, width - block_size)

    array[start_row:start_row + block_size, start_col:start_col + block_size] = value

    return array


# check_overlap remains the same as it needs start/size logic internally
def check_overlap(block1_start, block1_size, block2_start, block2_size):
    """Checks if two N-dimensional blocks overlap."""
    block1_end = block1_start + block1_size
    block2_end = block2_start + block2_size
    for i in range(len(block1_start)):
        if not (block1_start[i] < block2_end[i] and block2_start[i] < block1_end[i]):
            return False
    return True

def create_non_overlapping_blocks(
    dimension: tuple,
    n_blocks: int,
    max_block_size: tuple,
    min_block_size: tuple | None = None, # Added min_block_size
    max_attempts_per_block: int = 1000,
    seed: int | None = None
) -> list[tuple[slice, ...]]: # Return type is now list of tuples of slices
    """
    Creates a list of non-overlapping random block definitions usable for
    direct NumPy array indexing.

    Args:
        dimension (tuple): The dimensions of the container space (e.g., (height, width)).
                           Must have the same length as max_block_size.
        n_blocks (int): The number of blocks to attempt to create.
        max_block_size (tuple): The maximum size of a block along each dimension.
                                  Must have the same length as dimension.
                                  Each element must be >= 1.
        min_block_size (tuple | None, optional): The minimum size of a block
                                                 along each dimension. Must have
                                                 the same length as dimension.
                                                 Each element must be >= 1 and
                                                 <= corresponding max_block_size.
                                                 If None, defaults to (1, 1, ...).
                                                 Defaults to None.
        max_attempts_per_block (int): Maximum attempts to place a single block.
        seed (int | None, optional): Seed for the random number generator.

    Returns:
        list[tuple[slice, ...]]:
            A list where each element represents a successfully placed block.
            Each element is a tuple of Python `slice` objects, ready to be
            used for indexing a NumPy array representing the container space.
            Example for 2D: [(slice(y1, y1_end), slice(x1, x1_end)), ...]
            Returns fewer than n_blocks if placement fails repeatedly.

    Raises:
        ValueError: If dimension, max_block_size, or min_block_size (if provided)
                    have inconsistent lengths, or if size constraints are invalid
                    (e.g., min > max, min < 1, size > dimension).
    """
    dims = len(dimension)
    container_dim = np.array(dimension)
    max_b_size = np.array(max_block_size)

    # --- Input Validation ---
    if len(max_b_size) != dims:
        raise ValueError("Length of 'max_block_size' must match 'dimension'.")
    if not np.all(max_b_size >= 1):
        raise ValueError("All elements in 'max_block_size' must be >= 1.")

    # Process and validate min_block_size
    if min_block_size is None:
        # Default min size is 1 in all dimensions
        min_b_size = np.ones(dims, dtype=int)
    else:
        min_b_size = np.array(min_block_size)
        if len(min_b_size) != dims:
             raise ValueError("Length of 'min_block_size' must match 'dimension'.")
        if not np.all(min_b_size >= 1):
             raise ValueError("All elements in 'min_block_size' must be >= 1.")
        if not np.all(min_b_size <= max_b_size):
             raise ValueError("min_block_size must be <= max_block_size for all dimensions.")
        if not np.all(min_b_size <= container_dim):
             raise ValueError("min_block_size cannot be larger than the container dimension.")

    # Optional stricter check: max size cannot be larger than container
    if not np.all(max_b_size <= container_dim):
        print(f"Warning: Some max_block_size elements are larger than the corresponding container dimension.")
        # Adjust max_b_size to fit container if necessary for generation logic? Or rely on placement check.
        # Let's rely on placement check for now, but warn user. Max size check is primarily for sanity.

    # --- Initialization ---
    rng = np.random.default_rng(seed)
    # Store blocks internally using (start, size) for easier overlap checks
    internal_placed_blocks = []

    # --- Generation Loop ---
    for _ in range(n_blocks):
        placed_successfully = False
        for attempt in range(max_attempts_per_block):
            # 1. Generate random block size using min/max constraints
            # rng.integers uses [low, high) interval -> need max + 1
            block_size = rng.integers(min_b_size, max_b_size + 1, size=dims)

            # 2. Generate random starting position
            max_start_coords = container_dim - block_size
            max_start_coords = np.maximum(0, max_start_coords)
            upper_bounds = np.where(max_start_coords >= 0, max_start_coords + 1, 0)

            start_coords = np.zeros(dims, dtype=int)
            possible_to_generate = True
            for i in range(dims):
                if upper_bounds[i] <= 0: # Cannot generate if max_start is negative (block too big)
                     if block_size[i] > dimension[i]:
                          possible_to_generate = False
                          break
                     else: # If max_start is 0, only possible start is 0
                          start_coords[i] = 0
                else:
                     start_coords[i] = rng.integers(0, upper_bounds[i])

            if not possible_to_generate:
                continue # Try different size/position

            # 3. Check for overlap with previously placed blocks
            overlap = False
            # Use internal list [(start, size), ...] for check_overlap
            for existing_start, existing_size in internal_placed_blocks:
                if check_overlap(start_coords, block_size, existing_start, existing_size):
                    overlap = True
                    break

            # 4. If no overlap, store internally and mark success
            if not overlap:
                internal_placed_blocks.append((start_coords, block_size))
                placed_successfully = True
                break # Exit the attempt loop for this block

        if not placed_successfully:
            print(f"Warning: Failed to place block after {max_attempts_per_block} attempts. "
                  f"Returning {len(internal_placed_blocks)} blocks. Consider increasing attempts or reducing block density/size constraints.")
            # Stop trying for *this* specific block

    # --- Final Conversion to Slice Format ---
    output_slices = []
    for start, size in internal_placed_blocks:
        end = start + size
        # Create the tuple of slice objects
        slices = tuple(slice(start[d], end[d]) for d in range(dims))
        output_slices.append(slices)

    return output_slices


@dataclass
class LabelboxClassInstance:
    class_name: str
    class_idx: str
    rgb: tuple[int,int,int]

def generate_composite_mask_from_instances(width, height, instances:list[LabelboxClassInstance], min_block_size=50, max_block_size=50, seed=420):
    mask = np.zeros((height,width,3), dtype=np.uint8)
    blocks = create_non_overlapping_blocks(
        mask.shape[:2], 
        len(instances), 
        (max_block_size,max_block_size), 
        (min_block_size,min_block_size), 
        seed=seed
    )

    for block, instance in zip(blocks, instances):
        mask[block[0], block[1],:] = instance.rgb

    return mask
