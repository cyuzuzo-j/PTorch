
import json
import os

notebook_path = '/home/cyuzuzo/thesisV5/pjax-main/benchmark_conv.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Styles to inject
batch_size_code = [
    "plt.figure(figsize=(10, 5))\n",
    "\n",
    "plt.plot(batch_sizes, fft_times, 'b-o', label='FftConv2D')\n",
    "plt.plot(batch_sizes, regular_times, 'r-s', label='Conv2D')\n",
    "plt.title('Forward Pass Time vs Batch Size', fontsize=16)\n",
    "plt.xlabel('Batch Size', fontsize=12)\n",
    "plt.ylabel('Time (ms)', fontsize=12)\n",
    "plt.grid(True)\n",
    "plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.tight_layout()\n",
    "plt.savefig('batch_size_benchmark.png')\n",
    "plt.show()"
]

kernel_size_code = [
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(kernel_sizes, fft_times_kernel, 'b-o', label='FftConv2D', linewidth=2, markersize=8)\n",
    "plt.plot(kernel_sizes, regular_times_kernel, 'r-s', label='Conv2D', linewidth=2, markersize=8)\n",
    "plt.title('Forward Pass Time vs Kernel Size', fontsize=16)\n",
    "plt.xlabel('Kernel Size', fontsize=12)\n",
    "plt.ylabel('Time (ms)', fontsize=12)\n",
    "plt.grid(True)\n",
    "plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "plt.xticks(kernel_sizes)\n",
    "plt.tight_layout()\n",
    "plt.savefig('kernel_size_benchmark.png')\n",
    "plt.show()"
]

found_batch = False
found_kernel = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Identify Batch Size plotting cell
        if "plt.rcParams.update" in source and "batch_sizes" in source:
            print("Found Batch Size plotting cell.")
            cell['source'] = batch_size_code
            found_batch = True
            
        # Identify Kernel Size plotting cell
        elif "kernel_sizes" in source and "plt.plot" in source:
            # We need to distinguish it from the computation cell?
            # The plotting cell starts with plt.figure usually or has plt.show at the end
            if "plt.show()" in source:
                print("Found Kernel Size plotting cell.")
                cell['source'] = kernel_size_code
                found_kernel = True

if found_batch and found_kernel:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated notebook.")
else:
    print(f"Error: Could not find all cells. Batch: {found_batch}, Kernel: {found_kernel}")
