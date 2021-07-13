# Note:
# --feature_type can be set to DenseNet121, VGG19, or ResNet50.

# genetate annotation csv
python kitti-data/generate-csv.py --input=kitti-data/training/label_2/ --output=kitti-data/annotations.csv

# extract feature 
python feature_extraction.py --model VGG19 


# test CL_CSEN: 
python regressor_main.py --method CL-CSEN --feature_type VGG19  --weights True
