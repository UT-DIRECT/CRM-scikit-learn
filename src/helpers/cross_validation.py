from .analysis import fit_statistics

def forward_walk_and_ML(X, y, step_size, model):
    split_data = forward_walk(X, y, step_size)
    return train_and_test_model(model, split_data)

def forward_walk(X, y, step_size):
    length = len(X)
    split_data = {'x_train': [], 'x_test': [], 'y_train': [], 'y_test': []}
    for i in range(length - 2):
        train_end = i + 1
        test_start = i + 1
        test_end = (int(test_start + step_size)
            if test_start + step_size < length
            else length)
        
        x_train, x_test = X[:train_end], X[test_start: test_end]
        y_train, y_test = y[:train_end], y[test_start: test_end]
        split_data['x_train'].append(x_train)
        split_data['x_test'].append(x_test)
        split_data['y_train'].append(y_train)
        split_data['y_test'].append(y_test)
    return split_data

def train_and_test_model(model, split_data):
    r2_sum, mse_sum = 0, 0
    length = len(split_data['x_train'])
    for i in range(length):
        x_train, x_test = split_data['x_train'][i], split_data['x_test'][i]
        y_train, y_test = split_data['y_train'][i], split_data['y_test'][i]
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2_i, mse_i = fit_statistics(y_predict, y_test)
        r2_sum += r2_i
        mse_sum += mse_i
    r2 = r2_sum / length
    mse = mse_sum / length
    return (r2, mse)
