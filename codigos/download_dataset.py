import kagglehub


# Download latest version
path = kagglehub.dataset_download("muratkokludataset/dry-bean-dataset", output_dir="dataset", force_download=True)

print("Path to dataset files:", path)