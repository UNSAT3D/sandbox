import h5py
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def plot_image(data, index, name):
    # Get the specified image data
    img_data = data[index]

    # Normalize the data to the range [0, 1]
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    # Plot the image
    plt.imshow(img_data, cmap='gray')
    plt.title(name)
    plt.savefig(f"{name}.png")

def h5_to_numpy(h5_file):
    # Load the .h5 file
    with h5py.File(h5_file, 'r') as f:
        # Extract data from the 'OutArray' dataset into a numpy array
        data = np.array(f['OutArray'])
    return data

def tif_to_numpy(tif_file):
    images = []
    with Image.open(tif_file) as img:
        i = 0
        while True:
            try:
                img.seek(i)
                images.append(np.array(img))
                i += 1
            except EOFError:
                # Reached end of image sequence (file)
                break

    images = np.array(images)
    return images

def compute_stats(array):
    array_min = np.min(array)
    array_max = np.max(array)
    array_mean = np.mean(array)

    array_normalized = (array - array_min) / (array_max - array_min)
    array_std = np.std(array_normalized)

    stats_dict = {
        'min': array_min,
        'max': array_max,
        'mean': array_mean,
        'std (after normalization)': array_std
    }
    for k, v in stats_dict.items():
        print(f"{k}: {v}")
    return stats_dict

def normalize(array, stats):
    return (array - stats['min']) / (stats['max'] - stats['min'])

def main(sim_dir, exp_dir):
    sim_file = f"{sim_dir}/colour_output_t00250000-0285621391.h5"
    sim_array = h5_to_numpy(sim_file)

    exp_files = [f for f in os.listdir(exp_dir) if f.endswith('.tif')]
    exp_arrays = {f: tif_to_numpy(os.path.join(exp_dir, f)) for f in exp_files}

    print(f"Shapes: ")
    print(f"Simulation: {sim_array.shape}")
    print(f"Experimental: ")
    for f in exp_files:
        print(f"{f}: {exp_arrays[f].shape}")

    print("Statistics: ")
    sim_stats = compute_stats(sim_array)
    exp_stats = {}
    for f in exp_files:
        print(f"{f}: ")
        exp_stats[f] = compute_stats(exp_arrays[f])

    # normalized data
    sim_array_norm = normalize(sim_array, sim_stats)
    exp_arrays_norm = {f: normalize(exp_arrays[f], exp_stats[f]) for f in exp_files}

    # select patch of same size as simulation, taken from middle
    sim_w, sim_h = sim_array.shape[1], sim_array.shape[2]
    exp_w, exp_h = exp_arrays[exp_files[0]].shape[1], exp_arrays[exp_files[0]].shape[2]
    exp_w_start = int((exp_w - sim_w) / 2)
    exp_h_start = int((exp_h - sim_h) / 2)
    exp_arrays_patch = {f: exp_arrays_norm[f][:, exp_w_start:exp_w_start+sim_w, exp_h_start:exp_h_start+sim_h] for f in exp_files}

    # plot middle normalized images
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_image(sim_array_norm, 80, f"{plot_dir}/sim_norm")
    for f in exp_files:
        plot_image(exp_arrays_patch[f], 800, f"{plot_dir}/exp_norm_{f}")


if __name__ == '__main__':
    sim_dir = '../data/sim/'
    exp_dir = '../data/exp/'
    main(sim_dir, exp_dir)
