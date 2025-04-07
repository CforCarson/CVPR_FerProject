# Paths
FER2013_CSV_PATH = './data/fer2013/fer2013.csv'
SYNTHETIC_DATA_DIR = './data/synthetic/'
OUTPUT_DIR = './output'

# GAN Training Parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_CLS = 10.0
LAMBDA_TEX = 5.0

# ViT Training Parameters
VIT_EPOCHS = 50
VIT_LR = 0.0001

# Model Architecture Parameters
LATENT_DIM = 128
NUM_CLASSES = 7
EMBED_DIM = 256
NUM_HEADS = 8
PATCH_SIZE = 4

# Texture Analysis Parameters
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = 'uniform' 