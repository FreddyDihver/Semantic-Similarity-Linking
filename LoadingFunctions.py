import pandas as pd
def load_data(file_path, rs=42):
    """
    Reads a csv file and returns two dataframes: o (for original sentences), t (for target sentences)

    Returns two dataframes: o (for original sentences), t (for target sentences) 
    
    :param file_path: path to csv file
    :param rs: random state
    :return: two pandas DataFrames, o and t
    """
    # Read the csv file
    df = pd.read_csv(file_path)

    # Split the data by label
    # m: dataframe with label 1 (links)
    # nm: dataframe with label 0 (non-links)
    m = df[df["label"] == 1]
    nm = df[df["label"] == 0]

    print('Total links in dataset: ', len(m))
    print('Total non-links in dataset: ', len(nm))

    # Get the columns for the original and target dataframes
    cols_o = ["label"] + [c for c in df.columns if c.endswith("_o")]
    cols_t = ["label"] + [c for c in df.columns if c.endswith("_t")]

    # Return the two dataframes
    return df[cols_o], df[cols_t]


def load_unbal_train_eval(file_path, Nsamples=None, rs=42, trainsplit=0.8):
    """
    Returns two dataframes: o (for original sentences), t (for target sentences)

    :param file_path: path to csv file
    :param Nsamples: total number of samples (train + eval), unbalanced between matches and non-matches
    :param rs: random state for sampling and shuffling
    :param trainsplit: proportion of samples to use for training
    """
    df = pd.read_csv(file_path)

    if Nsamples is None:
        N = len(df)
    else:
        N = Nsamples

    # Split the data by label
    m = df[df["label"] == 1]  # matches
    nm = df[df["label"] == 0]  # non-matches

    print('Total links in dataset: ', len(m))
    print('Total non-links in dataset: ', len(nm))

    # Calculate the ratio of matches to non-matches
    m_mn_ratio = len(m) / len(nm)

    # Calculate the number of samples to use for training and evaluation
    trainN = int(N * trainsplit)
    evalN = N - trainN

    # Calculate the number of matches and non-matches for training and evaluation
    train_mN = int(trainN * m_mn_ratio)
    train_nmN = trainN - train_mN

    # Sample the training data
    trainsamp = pd.concat([m[:train_mN], nm[:train_nmN]], ignore_index=True).sample(frac=1, random_state=rs,
                                                                                    ignore_index=True)

    # Sample the evaluation data
    evalsamp = pd.concat([m[train_mN:], nm[train_nmN:]], ignore_index=True).sample(frac=1, random_state=rs,
                                                                                   ignore_index=True)

    if Nsamples is not None:
        trainsamp = trainsamp[:trainN]
        evalsamp = evalsamp[:evalN]

    # Get the columns for the original and target dataframes
    cols_o = ["label"] + [c for c in df.columns if c.endswith("_o")]
    cols_t = ["label"] + ["pa_id_o"] + [c for c in df.columns if c.endswith("_t")]

    # Get the original, target, labels, evaluation original, evaluation target, evaluation labels
    o = trainsamp[cols_o]
    t = trainsamp[cols_t]
    l = o["label"].values
    o_eval = evalsamp[cols_o]
    t_eval = evalsamp[cols_t]
    l_eval = o_eval["label"].values

    return o, t, l, o_eval, t_eval, l_eval

def load_bal_train_unbal_eval(file_path, Nsamples, rs=42, trainsplit=1.0, eval_sample_only=True):
    """
    Returns 6 dataframes, structered as: o(training original), t(training target), labels, o_eval(evalutation original), t_eval(evaluation target), evaluation labels

    :param file_path: path to csv file
    :param Nsamples: total number of samples (train + eval), balanced between matches and non-matchs for training data (but not eval data)
    :param rs: random state for sampling and shuffling
    :param trainsplit: split between training and evaluation  (only if eval_set_on=True)
    :param eval_sample_only: If false, takes all remaining data for evaluation sets (overriding trainsplit)
    """
    df = pd.read_csv(file_path)

    # Split by label
    m = df[df["label"] == 1]  # matches
    nm = df[df["label"] == 0]  # non-matches

    print(f'Total links in dataset: {len(m)}')
    print(f'Total non-links in dataset: {len(nm)}')

    if eval_sample_only:
        # Calculate the number of samples to use for training and evaluation
        m_mn_ratio = len(m) / len(nm)
        trainN = int(Nsamples * trainsplit)
        evalN = Nsamples - trainN

        # Calculate the number of matches and non-matches for training and evaluation
        train_mN = trainN // 2
        train_nmN = trainN - train_mN
        eval_mN = int(round(evalN * m_mn_ratio))
        eval_nmN = evalN - eval_mN

        # Check if there are enough rows for the requested Nsamples/trainsplit
        if train_mN + eval_mN > len(m) or train_nmN + eval_nmN > len(nm):
            need_m = train_mN + eval_mN
            need_nm = train_nmN + eval_nmN
            avail_m = len(m)
            avail_nm = len(nm)

            print(f"Requested positives: {need_m}, available positives: {avail_m}")
            print(f"Requested negatives: {need_nm}, available negatives: {avail_nm}")
            raise ValueError("Not enough rows for requested Nsamples/trainsplit")

        # Sample the training data
        m_samp = m.sample(train_mN + eval_mN, random_state=rs)
        nm_samp = nm.sample(train_nmN + eval_nmN, random_state=rs)

        # Sample the evaluation data
        trainsamp = pd.concat([m_samp[:train_mN], nm_samp[:train_nmN]], ignore_index=True).sample(frac=1,
                                                                                                  random_state=rs,
                                                                                                  ignore_index=True)
        evalsamp = pd.concat([m_samp[train_mN:], nm_samp[train_nmN:]], ignore_index=True).sample(frac=1,
                                                                                                 random_state=rs,
                                                                                                 ignore_index=True)

    else:
        # Sample the training data
        m_samp = m.sample(frac=1, random_state=rs)
        nm_samp = nm.sample(frac=1, random_state=rs)

        # Split the data into training and evaluation sets
        trainN = int(Nsamples * trainsplit)
        train_mN = trainN // 2
        train_nmN = trainN - train_mN

        trainsamp = pd.concat([m_samp[:train_mN], nm_samp[:train_nmN]], ignore_index=True).sample(frac=1,
                                                                                                  random_state=rs,
                                                                                                  ignore_index=True)
        evalsamp = pd.concat([m_samp[train_mN:], nm_samp[train_nmN:]], ignore_index=True).sample(frac=1,
                                                                                                 random_state=rs,
                                                                                                 ignore_index=True)

    cols_o = ["label"] + [c for c in df.columns if c.endswith("_o")]
    cols_t = ["label"] + ["pa_id_o"] + [c for c in df.columns if c.endswith("_t")]

    o = trainsamp[cols_o]
    t = trainsamp[cols_t]
    l = o["label"].values
    o_eval = evalsamp[cols_o]
    t_eval = evalsamp[cols_t]
    l_eval = o_eval["label"].values

    return o, t, l, o_eval, t_eval, l_eval