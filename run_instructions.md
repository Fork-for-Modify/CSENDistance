# Run Instructions
## data preprocess

### generate csv (can be skipped when testing - generate csv file manually)

``` python
python ./data/orig-data/generate-csv.py --input=./data/orig-data/test-data/chasing/label/ --output=./data/orig-data/test-data/chasing/annotations.csv
```

### generate feature

``` python
python feature_extraction.py --model VGG19 --imgdir ./data/orig-data/test-data/chasing/image/ --imgsuf .png --savedir ./data/features/test-data/
```

### feature processing

``` python
matlab processTestFeatures.m
```

## CL-CSEN

### train CL-CSEN

`python regressor_main.py --method CL-CSEN --feature_type VGG19  --datadir "./data/CSENdata-2D/demo/" --isTrain`

### test CL-CSEN

`python regressor_main.py --method CL-CSEN --feature_type VGG19  --datadir "./data/CSENdata-2D/test/" --savedir "./results/test/"`
