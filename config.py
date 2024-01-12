class Config:
    data_path = 'data/processed_data'
    train_data = 'data/processed_data/train_data.csv'
    train_label = 'data/processed_data/train_label.csv'
    test_data = 'data/processed_data/test_data.csv'
    test_label = 'data/processed_data/test_label.csv'
    val_data = 'data/processed_data/val_data.csv'
    val_label = 'data/processed_data/val_label.csv'

    pretrain_path = 'model_hub/chinese_wwm_ext_pytorch'

    checkpoints_path = 'checkpoints/model.pth'

    use_cuda = True
    batch_size = 4
    num_epochs = 3
    dropout = 0.5
    bert_lr = 1e-5
    base_lr = 1e-3

    max_length = 512
