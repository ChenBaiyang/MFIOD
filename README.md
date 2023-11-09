# MFIOD
Codes and datasets for the paper "Fusing multi-scale fuzzy information to detect outliers" accepted by Information Fusion (In Press).

## Datasets
We use the following 15 public datasets (https://github.com/BELLoney/Outlier-detection) to evaluate the detection method. Down-sampling method which randomly removes some objects in a particular class is adopted to produce an outlier set. Next, the maximum probability method is taken to handle the missing values, i.e., to fill in the blanks with the most frequently occurring values on other samples. In addition, all attribute values are transformed into the interval of [0,1] by adopting the min-max normalization. 

| No. |                Dataset               | Abbr. | #Objects | #Attributes | #Outliers | %Outlier | Data type |
|:---:|:------------------------------------:|:-----:|:--------:|:-----------:|:---------:|:--------:|:---------:|
|  1  |              Annthyroid              |  Ann  |   7200   |      6      |    534    |   7.4%   | Numerical |
|  2  |  Cardiotocography_2and3_33_variant1  |  Card |   1688   |      21     |     33    |   2.0%   | Numerical |
|  3  | Diabetes_tested_positive_26_variant1 |  Diab |    526   |      8      |     26    |   4.9%   | Numerical |
|  4  |                 Ecoli                | Ecoli |    336   |      7      |     9     |   2.7%   | Numerical |
|  5  |         German_1_14_variant1         |  Germ |    714   |      20     |     14    |   2.0%   |   Hybrid  |
|  6  |        Heart270_2_16_variant1        | Heart |    166   |      13     |     16    |   9.6%   |   Hybrid  |
|  7  |        Hepatitis_2_9_variant1        |  Hepa |    94    |      19     |     9     |   9.6%   |   Hybrid  |
|  8  |       Ionosphere_b_24_variant1       |  Iono |    249   |      34     |     24    |   9.6%   | Numerical |
|  9  |             Lymphography             |  Lymp |    148   |      18     |     6     |   4.1%   |  Nominal  |
|  10 |       Pageblocks_1_258_variant1      |  Page |   5171   |      10     |    258    |   5.0%   | Numerical |
|  11 |         Pima_TRUE_55_variant1        |  Pima |    555   |      9      |     55    |   9.9%   | Numerical |
|  12 |          Sonar_M_10_variant1         | Sonar |    107   |      60     |     10    |   9.3%   | Numerical |
|  13 |       Wbc_malignant_39_variant1      |  Wbc  |    483   |      9      |     39    |   8.1%   | Numerical |
|  14 |          Wdbc_M_39_variant1          |  Wdbc |    396   |      31     |     39    |   9.8%   | Numerical |
|  15 |         Yeast_ERL_5_variant1         | Yeast |   1141   |      8      |     5     |   0.4%   | Numerical |


## Environment
* python=3.8
* numpy=1.23
* scikit-learn=1.2
