# IEMOCAP
IEMOCAP_train_video_path = '../data/IEMOCAP/train/video'
IEMOCAP_train_text_path = '../data/IEMOCAP/train/text'
IEMOCAP_train_audio_path = '../data/IEMOCAP/train/audio'

IEMOCAP_valid_video_path = '../data/IEMOCAP/valid/video'
IEMOCAP_valid_text_path = '../data/IEMOCAP/valid/text'
IEMOCAP_valid_audio_path = '../data/IEMOCAP/valid/audio'

IEMOCAP_test_video_path = '../data/IEMOCAP/test/video'
IEMOCAP_test_text_path = '../data/IEMOCAP/test/text'
IEMOCAP_test_audio_path = '../data/IEMOCAP/test/audio'

IEMOCAP_train = '../data/IEMOCAP/train.txt'
IEMOCAP_valid = '../data/IEMOCAP/valid.txt'
IEMOCAP_test = '../data/IEMOCAP/test.txt'


# Training parameters
batch_size = 16
num_workers = 4
lr = 5e-5
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
Projection_dropout = 0.2
projection_dim = 256

# cls操作
tCLS_dropout = 0.3
tCLS_layers = 2
processor_dropout = 0.3
processor_hidden_dims = [512]

classifier_hidden_dims = [512, 256, 128]
classifier_dropout = 0.3
classifier_num_classes = 6                      ##### IEMOCAP

# num_channels = [768, 768, 768]
num_channels = [1024, 1024, 1024]
TNC_dropout = 0.3

similarity_threshold=0.6
VSC_alpha=0.9
temperature = 0.5
alpha = 0.5   #对比学习
beta  = 1.0   #交叉熵


