# Data paths
FER2013_DIR = './data/FER2013'
OUTPUT_DIR = './output'

# Generator parameters
LATENT_DIM = 128
GEN_EMBED_DIM = 256
GEN_NUM_HEADS = 8

# Training parameters
BATCH_SIZE = 64
GAN_EPOCHS = 100
VIT_EPOCHS = 50
GAN_LR = 0.0002
VIT_LR = 0.0001
BETA1 = 0.5
BETA2 = 0.999

# Loss weights
LAMBDA_CLS = 10.0  # Weight for classification loss
LAMBDA_TEX = 5.0   # Weight for texture preservation loss

# Evaluation parameters
NUM_SYNTHETIC_SAMPLES = 3500  # 500 per class for 7 classes