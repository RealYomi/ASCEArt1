
# %%
"""
Chat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def varioTry1T2(x, y, z):
    """
    Calculates and plots experimental variograms.
    
    Inputs:
      x : X-coordinates (1D list or array)
      y : Y-coordinates (1D list or array)
      z : Variable values (if on a grid, z should have the same dimensions as produced by meshgrid(x,y))
      
    The function creates a meshgrid from x and y, computes the upper‐triangular pairwise distances 
    and angles (in degrees), then aggregates the squared differences into lag bins and directions.
    
    It then plots both the structured (directional) and unstructured (isotropic) variograms.
    
    Returns:
      results : A NumPy array containing variogram results with header rows.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    # --- Create a meshgrid of the coordinates and flatten (mimicking Matlab’s [x,y] = meshgrid(x,y) then reshape) ---
    X, Y = np.meshgrid(x, y)
    x = X.ravel()
    y = Y.ravel()
    z = z.ravel()
    n = len(x)
    
    # --- Calculate pairwise distances (using broadcasting) ---
    # Compute full matrix of distances:
    d = np.sqrt((x.reshape(-1, 1) - x.reshape(1, -1))**2 +
                (y.reshape(-1, 1) - y.reshape(1, -1))**2)
    # Only keep the upper triangular part (as in Matlab’s triu)
    d = np.triu(d)
    
    # --- Compute angles (in degrees) between points ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ang_full = np.degrees(np.arctan((y.reshape(-1, 1) - y.reshape(1, -1)) /
                                        (x.reshape(-1, 1) - x.reshape(1, -1))))
    ang = np.triu(ang_full)
    
    # --- Get variogram parameters ---
    # Find the minimum positive values of x and y
    xPos = x[x > 0]
    yPos = y[y > 0]
    xMin = np.min(xPos) if xPos.size > 0 else 0
    yMin = np.min(yPos) if yPos.size > 0 else 0
    hMin = xMin * yMin  # minimum lag step
    
    # Set default variogram parameters:
    h    = hMin
    tol  = 0.5
    nDir = 1
    angTol = 0.5
    inAng = 0

    # --- Ask the user for parameters (if nothing is entered, the default is used) ---
    try:
        tmp = input(f"Enter lag steps (default {h}): ")
        h = float(tmp) if tmp.strip() != "" and float(tmp) > 0 else hMin
    except:
        h = hMin

    try:
        tmp = input(f"Enter the lag tolerance [0 - 0.5] (default {tol}): ")
        tol = float(tmp) if tmp.strip() != "" and 0 < float(tmp) <= 0.5 else 0.5
    except:
        tol = 0.5

    try:
        tmp = input(f"Enter the number of directions (default {nDir}): ")
        nDir = int(tmp) if tmp.strip() != "" and int(tmp) > 0 else 1
    except:
        nDir = 1

    try:
        tmp = input(f"Enter the direction tolerance (default {angTol}): ")
        angTol = float(tmp) if tmp.strip() != "" and float(tmp) > 0 else 0.1
    except:
        angTol = 0.1

    try:
        tmp = input(f"Enter the initial direction (default {inAng}): ")
        inAng = float(tmp) if tmp.strip() != "" else 0
    except:
        inAng = 0

    # --- Compute experimental variogram data ---
    # Get indices (row, col) where distance is nonzero (i.e. upper-triangular, excluding zeros)
    row_idx, col_idx = np.nonzero(d)
    distances = d[row_idx, col_idx]
    angles_vals = ang[row_idx, col_idx]
    
    # Compute differences and squared differences
    differences = z[row_idx] - z[col_idx]
    sq_diff = differences**2
    
    # Combine into a data array with columns:
    # [distance, row index, col index, difference, squared difference, angle]
    int_variogram_data = np.column_stack((distances, row_idx, col_idx, differences, sq_diff, angles_vals))
    
    # Sort the rows by distance (first column)
    sorted_int_variogram_data = int_variogram_data[np.argsort(int_variogram_data[:, 0])]
    sortedDis = sorted_int_variogram_data[:, 0].copy()
    
    # --- Aggregate distances that are close ---
    for i in range(len(sortedDis)-2, -1, -1):
        if abs(sortedDis[i+1] - sortedDis[i]) <= tol:
            sortedDis[i] = max(sortedDis[i], sortedDis[i+1])
    # Update the distance column in our data
    sorted_int_variogram_data[:, 0] = sortedDis
    
    # --- Aggregate angles within tolerance ---
    angDf = 180 / nDir
    dir_array = np.array([inAng + angDf * i for i in range(nDir)])
    vaAng = sorted_int_variogram_data[:, 5].copy()
    # Convert negative angles to positive (add 180)
    vaAng[vaAng < 0] += 180
    # For each preset direction, if the difference is within tolerance, assign it.
    for k in range(nDir):
        for i in range(len(vaAng)):
            if abs(vaAng[i] - dir_array[k]) <= angTol:
                vaAng[i] = dir_array[k]
    # Replace angles equal to 180 with 0.
    vaAng[vaAng == 180] = 0
    sorted_int_variogram_data[:, 5] = vaAng
    
    # --- Structured experimental variogram ---
    unique_sortedDis = np.unique(sortedDis)
    # Define plotting symbols similar to Matlab
    simb = ['r+-','k*-','b*-','gs-','mx-','co-']
    space = np.ones(5) * 1000000000
    results_list = []
    
    plt.figure()
    
    for j in range(nDir):
        summed_data_list = []
        counts_list = []
        std_list = []
        for d_val in unique_sortedDis:
            # Select rows where the binned distance equals d_val and the angle equals the current direction.
            mask = (np.isclose(sorted_int_variogram_data[:, 0], d_val)) & \
                   (np.isclose(sorted_int_variogram_data[:, 5], dir_array[j]))
            s = np.sum(sorted_int_variogram_data[mask, 4])
            c = np.sum(mask)
            std_val = np.std(sorted_int_variogram_data[mask, 4], ddof=0) if c > 0 else np.nan
            summed_data_list.append(s)
            counts_list.append(c)
            std_list.append(std_val)
        summed_data_arr = np.array(summed_data_list)
        counts_arr = np.array(counts_list, dtype=float)
        std_arr = np.array(std_list)
        with np.errstate(divide='ignore', invalid='ignore'):
            vario = 0.5 * (summed_data_arr / counts_arr)
        # Build a variogram data array with columns:
        # [unique distance, semivariance, summed squared diff, count, standard deviation]
        vario_data = np.column_stack((unique_sortedDis, vario, summed_data_arr, counts_arr, std_arr))
        # Remove rows with NaN semivariance
        vario_data = vario_data[~np.isnan(vario_data[:, 1])]
        
        # Prepend a header row: first element is the direction, the rest are 1e9.
        head = space.copy()
        head[0] = dir_array[j]
        results_list.append(head)
        for row in vario_data:
            results_list.append(row)
        
        # Plot this structured variogram
        plt.plot(vario_data[:, 0], vario_data[:, 1],
                 simb[j % len(simb)], linewidth=1.5, label=str(dir_array[j]))
    
    # --- Unstructured (Isotropic) Experimental Variogram ---
    summed_data_list = []
    counts_list = []
    std_list = []
    for d_val in unique_sortedDis:
        mask = np.isclose(sorted_int_variogram_data[:, 0], d_val)
        s = np.sum(sorted_int_variogram_data[mask, 4])
        c = np.sum(mask)
        std_val = np.std(sorted_int_variogram_data[mask, 4], ddof=0) if c > 0 else np.nan
        summed_data_list.append(s)
        counts_list.append(c)
        std_list.append(std_val)
    summed_data_arr = np.array(summed_data_list)
    counts_arr = np.array(counts_list, dtype=float)
    std_arr = np.array(std_list)
    with np.errstate(divide='ignore', invalid='ignore'):
        vario_iso = 0.5 * (summed_data_arr / counts_arr)
    vario_iso_data = np.column_stack((unique_sortedDis, vario_iso))
    vario_iso_data = vario_iso_data[~np.isnan(vario_iso_data[:, 1])]
    
    # Append the unstructured data to results (with a header row)
    head = space.copy()
    results_list.append(head)
    unstruct_data = np.column_stack((unique_sortedDis, vario_iso,
                                      summed_data_arr, counts_arr, std_arr))
    for row in unstruct_data:
        results_list.append(row)
    
    # Plot the unstructured variogram
    plt.plot(vario_iso_data[:, 0], vario_iso_data[:, 1], 'co-',
             linewidth=1.5, label='Isotropic')
    plt.title('Unstructured & Isotropic Experimental Variogram Plot')
    plt.xlabel('Distance(h)')
    plt.ylabel('Gamma(Semivariogram)')
    plt.legend(loc='upper left')
    plt.show()
    
    results = np.array(results_list)
    return results

# =============================================================================
# Example of how to call the function:
# =============================================================================
if __name__ == '__main__':
    # Load data (adjust the file path as necessary)
    # For example, load or define your x, y, z arrays.
    xdata_path = r'C:\x_cor.xlsx'
    x = pd.read_excel(xdata_path, header=None)
    ydata_path = r'C:\y_cor.xlsx'
    y = pd.read_excel(ydata_path, header=None)
    zdata_path = r'C:\z_data.xlsx'
    z = pd.read_excel(zdata_path, header=None)

    print ('xyz is defined')
    results = varioTry1T2(x, y, z)
    print("Variogram results:")
    print(results)