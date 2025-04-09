# Data paths
FER2013_DIR = './data/FER2013'
OUTPUT_DIR = './output'

# Generator parameters
LATENT_DIM = 128
GEN_EMBED_DIM = 256
GEN_NUM_HEADS = 8

# Training parameters
BATCH_SIZE = 64
GAN_EPOCHS = 150
VIT_EPOCHS = 50
GAN_LR = 0.0002
COMPLEX_GAN_LR = 0.0001  # Lower learning rate for complex GAN
VIT_LR = 0.0001
BETA1 = 0.5
BETA2 = 0.999

# Loss weights
LAMBDA_CLS = 12.0
LAMBDA_TEX = 8.0

# Evaluation parameters
NUM_SYNTHETIC_SAMPLES = 5000

# Data augmentation parameters
USE_AUGMENTED_DATASET = True
NUM_AUGMENTATIONS = 3

# Texture enhancement parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Balanced sampling
USE_BALANCED_SAMPLING = True

# Face validation parameters
FACE_VALIDATION_THRESHOLD = 0.66  # Minimum confidence for valid face
FACE_VALIDATION_BATCH_SIZE = 32   # Batch size for face validation
USE_FACE_VALIDATION = True        # Whether to use face validation during training

# Complex GAN parameters
COMPLEX_GAN_NGF = 64  # Generator feature maps
COMPLEX_GAN_NDF = 64  # Discriminator feature maps