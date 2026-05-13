import kagglehub

# Download latest version
path = kagglehub.dataset_download("gabrielvanzandycke/deepsport-dataset")

print("Path to dataset files:", path)