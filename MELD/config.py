# MELD
MELD_train_video_path = '../data/MELD/train/video'
MELD_train_text_path = '../data/MELD/train/text'
MELD_train_audio_path = '../data/MELD/train/audio'

MELD_valid_video_path = '../data/MELD/valid/video'
MELD_valid_text_path = '../data/MELD/valid/text'
MELD_valid_audio_path = '../data/MELD/valid/audio'

MELD_test_video_path = '../data/MELD/test/video'
MELD_test_text_path = '../data/MELD/test/text'
MELD_test_audio_path = '../data/MELD/test/audio'

MELD_train = '../data/MELD/train.csv'
MELD_valid = '../data/MELD/valid.csv'
MELD_test = '../data/MELD/test.csv'


# Training parameters
batch_size = 16
num_workers = 0
# lr = 1e-5
lr = 1e-4 # MELD
weight_decay = 1e-5
patience = 4
factor = 0.5
epochs = 30

# Image encoder model configuration
image_input = 768
image_embedding = 1024

# Text encoder model configuration
text_embedding = 1024

# Audio encoder model configuration
audio_embedding = 768


# Projection head configuration (used for both image and text encoders)
num_projection_layers = 2
Projection_dropout = 0.4
projection_dim = 256


# cls操作
tCLS_dropout = 0.3
tCLS_layers = 2
processor_dropout = 0.4
processor_hidden_dims = [512]

classifier_hidden_dims = [512, 256, 128]
classifier_dropout = 0.4
classifier_num_classes = 7                    ##### MELD

# num_channels = [768, 768, 768]
num_channels = [1024, 1024, 1024]
TNC_dropout = 0.4
similarity_threshold=0.6
VSC_alpha=0.9
temperature = 0.5
alpha = 0.5   #对比学习
beta  = 1.0   #交叉熵


