# MFIOD
Codes and datasets for the paper "Fusing multi-scale fuzzy information to detect outliers" accepted by Information Fusion (In Press).

## Datasets
We use the following 15 public datasets (https://github.com/BELLoney/Outlier-detection) to evaluate the detection method. Down-sampling method which randomly removes some objects in a particular class is adopted to produce an outlier set. Next, the maximum probability method is taken to handle the missing values, i.e., to fill in the blanks with the most frequently occurring values on other samples. In addition, all attribute values are transformed into the interval of [0,1] by adopting the min-max normalization. 
    
## Environment
* python=3.8
* numpy=1.23
* scikit-learn=1.2
