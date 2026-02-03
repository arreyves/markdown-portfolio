# Create the environment with Python 3.11
conda create -n laterite_eval python=3.11 -y

# Activate the environment
conda activate laterite_eval

# Install core data science and geospatial tools
conda install -c conda-forge pandas numpy matplotlib seaborn geopandas openpyxl jupyter -y

# Install R-essentials if you plan to use R within Jupyter
conda install -c conda-forge r-essentials r-base -y

# Install the kernel creator
conda install -c conda-forge ipykernel -y

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=laterite_eval --display-name "Python (Laterite_Eval)" 

