# TriDy

### Requirements
Python packages:
- pyflagsercontain - see https://github.com/JasonPSmith/pyflagsercontain
- pyflagser - https://github.com/giotto-ai/pyflagser
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
Download the spike train data as a numpy file from INSERT (or use your own) and edit the "spike_trains_address" entry in the json file to point to the spike trains.

Edit any other entries in the json file that need changing for your required parameters.

Run with::
    python ./pipeline.py example.json

The results of the pipeline will be printed to a file names "./results/classification_accuracies_"

Note that the example adjacency matrix is large and loaded as a full matrix in the code, using 8GB of memory, as such your system will need more than 8GB of memory to run the pipeline on this dataset.
