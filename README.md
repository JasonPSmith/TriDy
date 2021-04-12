# TriDy
**Tri**bal Classification of Binary **Dy**namics

A pipeline for classifying binary dynamics on digraphs using tribes (or closed neighbourhoods), proposed in:

[An application of neighbourhoods in digraphs to the classification of binary dynamics](https://arxiv.org/)

Pedro Conceição, Dejan Govc, Jānis Lazovskis, Ran Levi, Henri Riihimäki, and Jason P. Smith


### Requirements
Python packages:
- pyflagsercontain - see [here](https://github.com/JasonPSmith/pyflagsercontain)
- pyflagser - see [here](https://github.com/giotto-ai/pyflagser)
- pandas
- numpy
- subprocess
- concurrent
- os
- sys
- json
- networkx
- scipy
- pickle
- time
- scipy

### To run
Download the spike train data from [here](https://zenodo.org/record/4290212/files/input_data.zip) (or use your own), extract the file into the data folder and then run
```r
    (cd data && python extract_data.py)
```

Edit any entries in the json file that need changing for your required parameters.

Run with
```r
    python ./pipeline.py example.json
```

The results of the pipeline will be printed to a file names "./results/classification_accuracies_<parameter>.json", where parameter is the featurisation parameter used.

Note that the example adjacency matrix is large and loaded as a full matrix in the code, using 8GB of memory, as such your system will need more than 8GB of memory to run the pipeline on this dataset.
