import torch
import multiple_ticker_models as mtm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = dict()

if __name__ == '__main__':
    pos_sequence_length = range(10, 60)
    pos_hidden_dims = range(20, 40, 4)
    pos_num_epochs = range(80, 200, 20)
    pos_lr = [0.005, 0.01, 0.015, 0.02]

    for sequence_length in pos_sequence_length:
        for hidden_dims in pos_hidden_dims:
            for num_epochs in pos_num_epochs:
                for lr in pos_lr:
                    pass


