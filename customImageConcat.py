
import numpy as np


import numpy as np

def pad_or_truncate(array, target_columns):
    """
    Adjusts the number of columns of the input array to match the target number of columns.
    If the array has fewer columns, it will be padded with zeros.
    If the array has more columns, it will be truncated.
    
    Parameters:
        array (numpy.ndarray): The input array.
        target_columns (int): The target number of columns.
        
    Returns:
        numpy.ndarray: The adjusted array.
    """
    current_columns = array.shape[1]

    if current_columns == target_columns:
        return array  # No change needed

    elif current_columns < target_columns:
        # Create an array of zeros with the required number of columns and matching dimensions
        padding = target_columns - current_columns
        zero_padding = 255*np.ones((array.shape[0], padding, array.shape[2]))  # Zero padding along the columns (axis 1)
        return np.concatenate([array, zero_padding], axis=1)

    else:
        # Truncate the array if it has more columns than needed
        return array[:, :target_columns, :]





def custom_concat(*arrays):
    """
    Concatenates n numpy arrays based on the following logic:
    - If 2 arrays, concatenate them horizontally (along axis 1).
    - If 3 arrays, concatenate the first two horizontally, then the third array below the first along axis 0.
    - If 4 arrays, concatenate the first two horizontally, then the second pair horizontally, and combine both along axis 0.
    - If 5 arrays, continue concatenating in pairs horizontally and stack vertically, padding odd arrays to match dimensions.
    - If 6 arrays, do the same, but with three horizontal concatenations and then stack vertically.
    - If 7-10 arrays, continue grouping them in pairs and stacking the results vertically.

    Parameters:
        arrays: Variable-length input of NumPy arrays (same shape).

    Returns:
        A single concatenated NumPy array.
    """
    n = len(arrays)

    if n < 2:
        raise ValueError("At least 2 arrays are required for concatenation.")
    
    # Padding function to match the number of columns
    # def pad_or_truncate(array, target_columns):
    #     current_columns = array.shape[1]
        
    #     if current_columns == target_columns:
    #         return array  # No change needed
    #     elif current_columns < target_columns:
    #         # Pad with zeros if there are fewer columns
    #         padding = target_columns - current_columns
    #         return np.pad(array, ((0, 0), (0, padding)), mode="constant", constant_values=0)
    #     else:
    #         # Truncate if there are more columns
    #         return array[:, :target_columns]
    
    if n == 2:
        # Concatenate directly along axis 1 (horizontally)
        return np.concatenate(arrays, axis=1)
    
    elif n == 3:
        # Concatenate first two arrays along axis 1, then third below the first along axis 0
        temp = np.concatenate([arrays[0], arrays[1]], axis=1)  # Concatenate horizontally first two arrays
        third_array_adjusted = pad_or_truncate(arrays[2], temp.shape[1])
        return np.concatenate([temp, third_array_adjusted], axis=0)
    
    elif n == 4:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Stack both results vertically
        return np.concatenate([temp1, temp2], axis=0)
    
    elif n == 5:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Adjust the 5th array (pad or truncate to match)
        temp3 = pad_or_truncate(arrays[4], temp1.shape[1])
        
        # Stack all three results vertically
        return np.concatenate([temp1, temp2, temp3], axis=0)
    
    elif n == 6:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Concatenate last two arrays horizontally
        temp3 = np.concatenate([arrays[4], arrays[5]], axis=1)
        
        # Stack all results vertically
        return np.concatenate([temp1, temp2, temp3], axis=0)
    
    elif n == 7:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp3 = np.concatenate([arrays[4], arrays[5]], axis=1)
        
        # Adjust the 7th array (pad or truncate to match)
        temp4 = pad_or_truncate(arrays[6], temp1.shape[1])
        
        # Stack all results vertically
        return np.concatenate([temp1, temp2, temp3, temp4], axis=0)
    
    elif n == 8:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp3 = np.concatenate([arrays[4], arrays[5]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp4 = np.concatenate([arrays[6], arrays[7]], axis=1)
        
        # Stack all results vertically
        return np.concatenate([temp1, temp2, temp3, temp4], axis=0)
    
    elif n == 9:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp3 = np.concatenate([arrays[4], arrays[5]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp4 = np.concatenate([arrays[6], arrays[7]], axis=1)
        
        # Adjust the 9th array (pad or truncate to match)
        temp5 = pad_or_truncate(arrays[8], temp1.shape[1])
        
        # Stack all results vertically
        return np.concatenate([temp1, temp2, temp3, temp4, temp5], axis=0)
    
    elif n == 10:
        # Concatenate first two arrays horizontally
        temp1 = np.concatenate([arrays[0], arrays[1]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp2 = np.concatenate([arrays[2], arrays[3]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp3 = np.concatenate([arrays[4], arrays[5]], axis=1)
        
        # Concatenate next two arrays horizontally
        temp4 = np.concatenate([arrays[6], arrays[7]], axis=1)
        
        # Concatenate last two arrays horizontally
        temp5 = np.concatenate([arrays[8], arrays[9]], axis=1)
        
        # Stack all results vertically
        return np.concatenate([temp1, temp2, temp3, temp4, temp5], axis=0)

# Example usage:
if __name__ == "__main__":
    # Create example arrays with the same shape (3x5 for simplicity)
    a = np.random.rand(3, 5)  # Shape (3, 5)
    b = np.random.rand(3, 5)  # Shape (3, 5)
    c = np.random.rand(3, 5)  # Shape (3, 5)
    d = np.random.rand(3, 5)  # Shape (3, 5)
    e = np.random.rand(3, 5)  # Shape (3, 5)
    f = np.random.rand(3, 5)  # Shape (3, 5)
    g = np.random.rand(3, 5)  # Shape (3, 5)
    h = np.random.rand(3, 5)  # Shape (3, 5)
    i = np.random.rand(3, 5)  # Shape (3, 5)
    j = np.random.rand(3, 5)  # Shape (3, 5)
    
    # Concatenate 10 arrays
    result = custom_concat(a, b, c, d, e, f, g, h, i, j)
    
    print("Result shape:", result.shape)
    print(result)




