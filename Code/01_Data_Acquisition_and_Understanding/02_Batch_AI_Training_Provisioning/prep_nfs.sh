sudo apt-get update
sudo apt-get install unzip
mkdir -p /data/training_images
mkdir -p /data/validation_images
wget https://mawahstorage.blob.core.windows.net/aerialimageclassification/imagesets/balanced_training_set.zip
wget https://mawahstorage.blob.core.windows.net/aerialimageclassification/imagesets/balanced_validation_set.zip
unzip balanced_validation_set.zip -d /data/validation_images
unzip balanced_training_set.zip -d /data/training_images
