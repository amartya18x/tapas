Here is the Pytorch code for the accuracy results of Table 3 in the paper.


Here's how to run each dataset:

# Cancer
```python main.py --data='cancer' --divs=3 --n_agg=1 --layers=1```

# Diabetes
```python main.py --data='diabetes' --divs=3 --n_agg=1 --layers=2 --hf=10```

# Faces
First you need to download the datasets here:

faces_train.npz: https://www.dropbox.com/s/txvyjlng48t6h1m/faces_train.npz?dl=0

faces_test.npz: https://www.dropbox.com/s/qbxdi442zjtefpm/faces_test.npz?dl=0

and then place them in data/faces. Then you can run:
```python main.py --data='faces' --n_agg=1 --layers=5```

# MNIST
```python main.py --data='mnist' --n_agg=1 --layers=4 --ir=2```


## Dropping
To add weight-dropping to any run add the flag:

--drop_rate=0.1

this will drop 10% of the weights.


# Raw Datasets
If for any reason you would like the raw datasets they are here (or if small enough in their data folders already)

diabetes: https://www.dropbox.com/s/h9z6mqkw6rslt00/diabetic_data.csv?dl=0

lfwa faces: https://www.dropbox.com/s/gnlc0yq4b7kvduz/lfwa.tar.gz?dl=0

