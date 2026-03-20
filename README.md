# InterpolAI

We recommend using a conda environment over a virtual environment.
## Installation Steps

1. **Create a Conda Environment with Python 3.9**
   ```bash
   conda create --name InterpolAI python=3.9
   ```

2. **Activate the Conda Environment**
   ```bash
   conda activate InterpolAI
   ```

3. **Install Required Packages**
   - **For macOS (M1/M2/M4 Pro Chip)**
     ```bash
     pip install -r requirements_macos.txt
     ```
     This command installs the necessary packages optimized for macOS, including TensorFlow for Apple Silicon.

   - **For Windows**
     ```bash
     conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
     ```
     
     ```bash
     pip install -r requirements.txt
     ``` 

## Alternative Installation Method
If you prefer using yml files for installation on windows machines, you can use the following commands:
```bash
conda env create -f environment_3090.yml
```
or 
```bash
conda env create -f environment_4090.yml
```
This will create a conda environment with the necessary dependencies for running the application on NVIDIA GPUs. The `environment_3090.yml` is optimized for RTX 3090, while `environment_4090.yml` is optimized for RTX 4090.

**Activate the environment**
```bash
conda activate InterpolAI
```
**MACOS machines**, you can use the following commands:
```bash
conda env create -f environment_macos.yml
```
## Activate the environment:
```bash
conda activate InterpolAI
```
## Model/weights download:
Please download the model folder from the following Google Drive link: [model](https://drive.google.com/drive/folders/16a4zhopq8AfKCADXxBwuYccGr_PnBRlt?usp=sharing)  
Once downloaded please place the model folder inside  the interpolation directory of the InterpolAI repository. 
## Usage

In the interpolation folder, you can find individual executable Jupyter notebooks as listed:

1. `interpolAI_auto.ipynb` : Detects missing images from filenames and generates them automatically.
   - **OR** run from the CLI with mode `auto`:
   ```bash
   python main.py --mode auto --tile_size 1024 1024 --pth /path/to/your/images
   ```
   Add `--output /path/to/output` to write results to a separate folder.

2. `interpolAI_no_skip.ipynb`: Generates a given number of intermediate frames between every consecutive image pair.
   - **OR** run from the CLI with mode `no_skip`:
   ```bash
   python main.py --mode no_skip --tile_size 1024 1024 --pth /path/to/your/images --skip 1 3 5
   ```

3. `interpolAI_skip_haralick.ipynb`: Skips images in the folder and generates the skipped frames.
   - **OR** run from the CLI with mode `skip`:
   ```bash
   python main.py --mode skip --tile_size 1024 1024 --pth /path/to/your/images --skip 1
   ```

### CLI reference

```
python main.py --mode {auto,no_skip,skip}
               --tile_size H W
               --pth PATH
               [--output OUTPUT_DIR]
               [--skip N [N ...]]
```

| Flag | Description |
|---|---|
| `--mode` | `auto` detects gaps; `no_skip` inserts frames between every pair; `skip` pairs images separated by a given distance |
| `--tile_size` | Tile size for large images (height width). Use 1024 1024 as a starting point. |
| `--pth` | Path to folder containing input images (.tif / .png / .jpg) |
| `--output` | Output root folder for generated subfolders (default: same as `--pth`) |
| `--skip` | One or more skip values (used by `no_skip` and `skip` modes) |