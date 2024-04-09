import torch
import multiple_ticker_models as mtm
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = dict()

if __name__ == '__main__':
    model_path = mtm.model_path

    pos_sequence_length = range(10, 60, 10)
    pos_hidden_dims = range(20, 40, 4)
    pos_num_epochs = range(80, 200, 20)
    pos_lr = [0.005, 0.01, 0.015, 0.02]
    num_layers = 2
    model_type = 'LSTM'

    training = mtm.training_tickers
    testing = mtm.testing_tickers

    best_mse = 0
    best_params = ()
    best_model = None

    with open(f"{model_path}/gridsearch.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_type', 'sequence_length', 'hidden_dims', 'num_layers', 'num_epochs', 'lr', 'score'])
        for sequence_length in pos_sequence_length:
            for hidden_dims in pos_hidden_dims:
                for num_epochs in pos_num_epochs:
                    for lr in pos_lr:
                        model, score = mtm.run_and_eval(training, testing, model_type, sequence_length, hidden_dims,
                                                        num_layers, num_epochs, lr)

                        writer.writerow([model_type, sequence_length, hidden_dims, num_layers, num_epochs, lr, score])
                        print(f"model_type: {model_type}, sequence_length: {sequence_length}, hidden_dims: {hidden_dims},"
                              f"num_epochs: {num_epochs}, learning rate: {lr}. SCORE: {score}")
                        if score > best_mse:
                            best_mse = score
                            best_params = (sequence_length, hidden_dims, num_epochs, lr)
                            best_model = model
                            model.save_model(f"{model_path}/grid_search_{model_type}_{sequence_length}_{hidden_dims}_{num_epochs}_{lr}.pt")



