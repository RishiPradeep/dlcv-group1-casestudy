import kagglehub

# Download latest version
path = kagglehub.model_download("google/cropnet/tfLite/classifier-cassava-disease-v1")

print("Path to model files:", path)