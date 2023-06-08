# 1. install the required enviroment
conda create --name Sauvola --file spec-env.txt
conda activate Sauvola
pip install tensorflow-gpu==2.3
pip install opencv-python
pip install pandas
pip install parse

# 
# 2. test run
SauvolaInferenceDemo.ipynb
