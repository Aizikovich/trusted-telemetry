import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datasets import DatasetLSTM


def load_csvs(test=False):
    def fix_ues_steps(df):
        fix_ = []
        for i in range(len(df['step'].unique())):
            fix_ += [i] * 50
        df['step'] = fix_
        return df

    ue0 = pd.read_csv('data/ue.csv')
    cell0 = pd.read_csv('data/cell.csv')
    fix_steps = []
    for i in range(60):
        fix_steps += [i, i, i, i, i, i]
    cell0['step'] = fix_steps

    ue3 = pd.read_csv('data/ue_norm_seed_3.csv')
    cell3 = pd.read_csv('data/cell_norm_seed_3.csv')

    ue4 = pd.read_csv('data/ue_norm_seed_4.csv')
    cell4 = pd.read_csv('data/cell_norm_seed_4.csv')

    ue5 = pd.read_csv('data/ue_norm_seed_5.csv')
    cell5 = pd.read_csv('data/cell_norm_seed_5.csv')

    ue6 = pd.read_csv('data/ue_norm_seed_6.csv')
    cell6 = pd.read_csv('data/cell_norm_seed_6.csv')

    ue7 = pd.read_csv('data/ue_norm_seed_7.csv')
    cell7 = pd.read_csv('data/cell_norm_seed_7.csv')

    ue70_42 = pd.read_csv('data/ue_norm70_seed_42.csv')
    cell70_42 = pd.read_csv('data/cell_norm70_seed_42.csv')

    ue70_43 = pd.read_csv('data/ue_norm70_seed_43.csv')
    cell70_43 = pd.read_csv('data/cell_norm70_seed_43.csv')
    ue70_43 = fix_ues_steps(ue70_43)

    ue100_norm44 = pd.read_csv('data/ue_norm100_seed_44.csv')
    cell100_norm44 = pd.read_csv('data/cell_norm100_seed_44.csv')
    ue100_norm44 = fix_ues_steps(ue100_norm44)
    # evaluation data
    ue8 = pd.read_csv('data/ue_norm_seed_8.csv')
    cell8 = pd.read_csv('data/cell_norm_seed_8.csv')

    fix_steps = []
    for i in range(62):
        fix_steps += [i] * 50

    ue_m = pd.read_csv('data/ue_mali5_2805.csv')
    cell_m = pd.read_csv('data/cell_mali5_2805.csv')
    ue_m['step'] = fix_steps

    ue70 = pd.read_csv('data/ue_norm70_seed_42.csv')
    cell70 = pd.read_csv('data/cell_norm70_seed_42.csv')

    ues = [ue70, ue0, ue3, ue4, ue5, ue6, ue7, ue70_42, ue70_43, ue100_norm44]
    cells = [cell70, cell0, cell3, cell4, cell5, cell6, cell7, cell70_42, cell70_43, cell100_norm44]

    if test:
        ues = [ue8, ue_m]
        cells = [cell8, cell_m]

    for i in range(len(ues)):
        ues[i]['ue-id'] = ues[i]['ue-id'].apply(lambda x: int(x[2:]))

    return ues, cells


def update_cols(ue, cell):
    ues_cols = [
        'DRB.UEThpDl', 'RF.serving.RSRP', 'RF.serving.RSSINR',
        'nrCellIdentity',
        'rsrp_nb0', 'rsrp_nb1', 'rsrp_nb2', 'rsrp_nb3', 'rsrp_nb4',
        'rssinr_nb0', 'rssinr_nb1', 'rssinr_nb2', 'rssinr_nb3', 'rssinr_nb4',
        'step', 'targetTput', 'x', 'y', 'ue-id']

    cells_cols = [
        'measPeriodPrb', 'nrCellIdentity', 'pdcpBytesDl',
        'pdcpBytesUl', 'step', 'throughput', 'x', 'y']

    ue = ue[ues_cols]
    cell = cell[cells_cols]
    return ue, cell


#
def marge_by_step(ue, cell, steps=60):
    dfs = []
    for s in range(steps):
        cell_s = cell[cell['step'] == s]
        ues_s = ue[ue['step'] == s]
        merged_s = pd.merge(ues_s, cell_s, on='nrCellIdentity', how='right')
        # merged_s.drop(columns=['step_x'], inplace=True)
        merged_s.drop(columns=['step_x'], inplace=True)
        # merged_s.rename(columns={'step_y': 'step'}, inplace=True)
        dfs.append(merged_s)
    return pd.concat(dfs, ignore_index=True)




def get_dfs(ues, cells):
    res = []
    metrics = ['measPeriodPrb', 'throughput', 'DRB.UEThpDl', 'RF.serving.RSRP', 'RF.serving.RSSINR', 'step_y',
               'nrCellIdentity', 'ue-id']
    for i in range(len(ues)):
        u, c = update_cols(ues[i], cells[i])
        steps = max(u['step'])
        curr = marge_by_step(u, c, steps=steps)[metrics].rename(
            columns={'step_y': 'step', 'RF.serving.RSRP': 'rsrp', 'RF.serving.RSSINR': 'rssinr'})
        res.append(curr)
    return res


def create_idxs(dfs, win_size=3, only_cell=False):
    lst = []
    win = win_size
    counter = 0
    max_cells = max(dfs[0]['nrCellIdentity']) + 1

    max_steps = [max(df['step']) for df in dfs]
    # print("max step", (max_steps))
    for s in range(max(max_steps)):
        for cell in range(1, max_cells + 1):
            if only_cell:
                if cell != only_cell:
                    continue
            for j, df in enumerate(dfs):
                for w in range(win):
                    if s + w > max_steps[j]:
                        continue
                    df_cell = df[df['nrCellIdentity'] == cell]
                    try:
                        df_cell_step = df_cell[df_cell['step'] == s + w]
                        df_cell_step['idx'] = [int(counter)] * df_cell_step.shape[0]
                        # if cell == 5 and counter == 16:
                        #     print(f"here step: {s + w}, cell: {cell}, idx: {counter}\n{df_cell_step}")

                        lst.append(df_cell_step)
                    except:
                        continue
                counter += 1
    concat_df = pd.concat(lst, ignore_index=True)
    concat_df['idx'] = concat_df['idx'].astype(int)
    return concat_df


def scale_df(df, test=False, scaler=None):
    # print(df)
    for_plot = df.copy()
    for_plot = for_plot[for_plot['nrCellIdentity'] == 3]
    data = df.drop(columns=['step', 'nrCellIdentity', 'idx'], inplace=False)
    if test == False:
        # print(data)
        print("fitting scaler")
        scaler = StandardScaler()
        scaler.fit(data)

    scale_data = scaler.transform(data)
    scaled_df = pd.DataFrame(scale_data, columns=data.columns)

    scaled_df['step'] = df['step']
    scaled_df['nrCellIdentity'] = df['nrCellIdentity']
    scaled_df['idx'] = df['idx']

    return scaled_df, scaler


def get_cell_only(df_lst):
    res = []
    for df in df_lst:
        res.append(df)
    return res


def features_extraction(dfs, only_cell=False, single_cell=False):
    for i in range(len(dfs)):
        ue_count = dfs[i].groupby(['step', 'nrCellIdentity']).count().reset_index()['DRB.UEThpDl']
        dfs[i] = dfs[i].groupby(['step', 'nrCellIdentity']).mean().reset_index()
        dfs[i]['ue_count'] = ue_count

    for i in range(len(dfs)):
        dfs[i] = dfs[i].sort_values(by=['nrCellIdentity', 'step'])
        dfs[i]['prev_ue_count'] = dfs[i].groupby('nrCellIdentity')['ue_count'].shift(1)
        dfs[i]['new_ues'] = dfs[i]['ue_count'] - dfs[i]['prev_ue_count']
        dfs[i]['new_ues'] = dfs[i]['new_ues'].fillna(0).astype(int)
        dfs[i] = dfs[i].drop(columns=['prev_ue_count'])
        dfs[i] = dfs[i].sort_values(by=['step', 'nrCellIdentity'])

    if only_cell:
        for i in range(len(dfs)):
            dfs[i] = dfs[i].drop(columns=['rsrp', 'rssinr', 'DRB.UEThpDl'])
    if single_cell != False:
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i]['nrCellIdentity'] == single_cell]
    # make all step columns as int type
    for i in range(len(dfs)):
        dfs[i]['step'] = dfs[i]['step'].astype(int)
    return dfs


def plot_loss(metrics, key, save=False, name='train_loss'):
    _, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_title(f"Train Loss {name}")
    ax.plot(metrics[key])
    plt.xlabel("epoch")
    plt.ylabel('loss')
    if save:
        plt.savefig(f'plots/{name}.png')
    plt.show()


def find_best_threshold(normal_loss, anomaly_loss, name='', save=False, verbose=False):
    thresholds = np.linspace(0, 4, 100)  # Range of threshold values to test
    best_threshold = 0
    best_f1_score = 0
    metrics = []

    for threshold in thresholds:
        tp = fp = tn = fn = 0

        for loss in normal_loss:
            if loss > threshold:
                fp += 1
            else:
                tn += 1

        for loss in anomaly_loss:
            if loss > threshold:
                tp += 1
            else:
                fn += 1

        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        balanced_accuracy = (recall + tn / (tn + fp)) / 2

        metrics.append((threshold, accuracy, precision, recall, f1_score, balanced_accuracy))

        # Check if this threshold gives the best F1 score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold
    # best_threshold =1.7
    # Print the best threshold and corresponding metrics
    if verbose:
        print(f"Best Threshold: {best_threshold:.2f}")
        print(f"Best F1 Score: {best_f1_score:.2f}")

    # Plot the metrics as a function of the threshold
    metrics = np.array(metrics)
    plt.figure(figsize=(10, 6))
    plt.plot(metrics[:, 0], metrics[:, 1], label='Accuracy')
    plt.plot(metrics[:, 0], metrics[:, 2], label='Precision')
    plt.plot(metrics[:, 0], metrics[:, 3], label='Recall')
    plt.plot(metrics[:, 0], metrics[:, 4], label='F1 Score')
    # plt.plot(metrics[:, 0], metrics[:, 5], label='Balanced Accuracy')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title(f'Performance Metrics Thresholds {name}')
    if save:
        plt.savefig(f'plots/{name}_performance_metrics_thresholds.png')
    plt.show()
    return best_threshold


def load_single_cell_models_and_data(data_frames, architectures, args, model_path, device):
    """
    :param data_frames:
    :param architectures:
    :param args: input_size, hidden_size, num_layers, dropout *in this order!!!*
    :param dataset:
    :return:
    """
    single_cell_models = []
    input_size, hidden_size, num_layers, dropout = args
    for i in range(1, 7):
        # model_name = f'cell{i}_only_ue_aggregated_autoencoder.pth'
        # print("loading model: ", model_name)
        model = architectures
        model.load_state_dict(torch.load(f'{model_path}', map_location=device))
        model.to(device)
        single_cell_models.append(model)
    print("models loaded")
    #  get data loaders for all cells trainng data

    cell_loaders = []
    for i in range(6):
        cell = data_frames[i]
        # print(cell.shape)
        dataset = DatasetLSTM(cell)
        # if i == 1:
        #     print(dataset[751])
        cell_loader = DataLoader(dataset, batch_size=1)
        cell_loaders.append(cell_loader)
    return single_cell_models, cell_loaders


def get_latent_space(single_cell_models, cell_loaders):
    latent_space = {}
    memo = set()
    for i, model in enumerate(single_cell_models):
        cell_loader = cell_loaders[i]
        # print(f'HERE cell_{i+1} len: {len(cell_loader)}')
        latent = []
        ct, ce = 0, 0
        model.eval()
        # if i == 1:
        #     print(cell_loader[751])
        with torch.no_grad():
            for bx, data in enumerate(cell_loader):
                try:
                    sample, h = model.encoder(data)
                    ct += 1
                except RuntimeError as e:
                    # print(f"\nbx: {bx}, cell_{i + 1}\n")
                    memo.add((i,bx))
                    ce += 1
                    continue
                latent.append((data, sample))
        print(f"cell_{i + 1} | len: {len(cell_loader)} | total: {ct}, error: {ce}")
        # print(f'cell_{i} latent space shape: {len(latent)}')
        latent_space[f'{i + 1}'] = latent
    print(memo)
    return latent_space


def plot_confusion_matrix(loss_dist_normal, loss_dist_anomaly, best_threshold, name='cell', save=False, verbose=True):
    tp = fp = tn = fn = 0
    for loss in loss_dist_normal:
        if loss > best_threshold:
            fp += 1
        else:
            tn += 1

    for i, loss in enumerate(loss_dist_anomaly):
        if loss > best_threshold:
            tp += 1
        else:
            fn += 1
            print("missed anomaly at index: ", i)

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center')

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    balanced_accuracy = (recall + tn / (tn + fp)) / 2
    print("TP: ", tp, "FP: ", fp, "TN: ", tn, "FN: ", fn)
    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Balanced Accuracy": balanced_accuracy
    }
    if verbose:
        print(result)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.yticks([0, 1], ['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix {name} at Threshold {best_threshold:.2f}')
    plt.grid(0)
    if save:
        plt.savefig(f'plots/{name}_confusion_matrix.png')
    plt.show()
    return result


def features_extraction_new(dfs, only_cell=False, single_cell=False):
    for i in range(len(dfs)):
        # Count UEs
        ue_count = dfs[i].groupby(['step', 'nrCellIdentity']).count().reset_index()['DRB.UEThpDl']
        new_ue_counts = []
        left_ue_counts = []

        for cell, group in dfs[i].groupby('nrCellIdentity'):
            # seen_ues = set()
            previous_step_ues = set()
            # print(group)
            for step, ues in group.groupby('step')['ue-id']:
                current_step_ues = set(ues)

                new_ues = current_step_ues - previous_step_ues
                left_ues = previous_step_ues - current_step_ues

                new_ue_counts.append({'step': step, 'nrCellIdentity': cell, 'new_ue_count': len(new_ues)})
                left_ue_counts.append({'step': step, 'nrCellIdentity': cell, 'left_ue_count': len(left_ues)})

                previous_step_ues = current_step_ues

        new_ue_df = pd.DataFrame(new_ue_counts)
        left_ue_df = pd.DataFrame(left_ue_counts)

        # Merge new and left UE counts back into the original dataframe
        dfs[i] = dfs[i].merge(new_ue_df, on=['step', 'nrCellIdentity'], how='left')
        dfs[i] = dfs[i].merge(left_ue_df, on=['step', 'nrCellIdentity'], how='left')

        cell_feats = ['step', 'nrCellIdentity', 'new_ue_count', 'left_ue_count', 'ue-id']
        # Calculate mean and std
        agg_funcs = {col: ['mean', 'std'] for col in dfs[i].columns if col not in cell_feats}
        grouped = dfs[i].groupby(['step', 'nrCellIdentity']).agg(agg_funcs).reset_index()

        # Flatten the MultiIndex columns
        grouped.columns = ['_'.join(col).rstrip('_') for col in grouped.columns]

        # Merge the new and left UE counts back to the grouped dataframe
        grouped = grouped.merge(new_ue_df, on=['step', 'nrCellIdentity'], how='left')
        grouped = grouped.merge(left_ue_df, on=['step', 'nrCellIdentity'], how='left')

        # Add the ue_count back
        grouped['ue_count'] = ue_count

        dfs[i] = grouped

        # Filter by single cell if specified
        if single_cell:
            dfs[i] = dfs[i][dfs[i]['nrCellIdentity'] == single_cell]

        # make all nrCellIdentity columns as int type
        dfs[i]['nrCellIdentity'] = dfs[i]['nrCellIdentity'].astype(int)

    return dfs


def features_extraction_cells(dfs):
    res = []

    def safe_len_set(x):
        # if {nan} return 0 else return len(x)
        if isinstance(x, set) and len(x) == 1 and any(isinstance(item, float) and np.isnan(item) for item in x):
            return 0
        return len(x)

    def calculate_ue_changes(group):
        ues_by_step = group.groupby('step')['ue-id'].apply(lambda x: {ue for ue in x})
        ue_count = ues_by_step.apply(safe_len_set)
        new_ues = ues_by_step - ues_by_step.shift(1)
        left_ues = ues_by_step.shift(1) - ues_by_step
        # replace NaNs with empty sets
        new_ues = new_ues.fillna("")
        left_ues = left_ues.fillna("")

        res = pd.DataFrame({
            'ue_count': ue_count,
            'new_ue_count': new_ues.apply(safe_len_set),
            'left_ue_count': left_ues.apply(safe_len_set)
        })
        # print(res)
        return res
        # Perform the aggregations

    for df_a in dfs:
        cells = []
        for cid in range(1, 7):
            df = df_a[df_a['nrCellIdentity'] == cid]
            result = df.groupby(['nrCellIdentity', 'step']).agg({
                'DRB.UEThpDl': ['mean', 'std'],
                'rsrp': ['mean', 'std'],
                'rssinr': ['mean', 'std'],
                'ue-id': lambda x: list(x),  # We'll use this to calculate UE changes
                'measPeriodPrb': 'first',  # Keep the first value for each group
                'throughput': 'first'  # Keep the first value for each group
            })

            # Flatten the column names
            result.columns = ['_'.join(col).strip() for col in result.columns.values]

            # Calculate UE changes
            ue_changes = df.groupby('nrCellIdentity').apply(calculate_ue_changes).reset_index()

            # Rename the columns
            ue_changes = ue_changes.rename(columns={'level_1': 'step'})

            # Merge the results
            final_result = pd.merge(result.reset_index(), ue_changes, on=['nrCellIdentity', 'step'])

            # Clean up the columns
            final_result = final_result.drop('ue-id_<lambda>', axis=1)
            final_result = fill_missing_values(final_result)

            # Rename the new columns to remove the '_first' suffix
            final_result = final_result.rename(columns={
                'measPeriodPrb_first': 'measPeriodPrb',
                'throughput_first': 'throughput'
            })
            cells.append(final_result)
        cells_cat = pd.concat(cells)
        res.append(cells_cat)
    # sort every df in res first by step and then by cell id
    for i in range(len(res)):
        res[i] = res[i].sort_values(by=['step', 'nrCellIdentity'])
    return res


def fill_missing_values(df):
    # fill null values with knn imputer not idx and step
    imputer = KNNImputer(n_neighbors=3)
    impute_df = df.drop(columns=['step', 'nrCellIdentity'])
    impute_df = pd.DataFrame(imputer.fit_transform(impute_df), columns=impute_df.columns)
    impute_df['step'] = df['step']
    impute_df['nrCellIdentity'] = df['nrCellIdentity']
    return impute_df
