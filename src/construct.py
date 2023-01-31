from sklearn.model_selection import StratifiedKFold, KFold


def construct_cv(cvtype='skf', nfolds=5, random_state=42, shuffle=True):
    """Construct cross-validation object.
    Args:
        cvtype (str): type of cross-validation (default: 'skf')
        nfolds (int): number of folds (default: 5)
        random_state (int): random state (default: 42)
        shuffle (bool): shuffle data (default: True)
    Returns:
        cv (object): cross-validation object
    """
    if cvtype == 'skf':
        cv = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=random_state)
    elif cvtype == 'kf':
        cv = KFold(n_splits=nfolds, shuffle=shuffle, random_state=random_state)
    else:
        raise ValueError(f"Unknown cross-validation type: {type}")
    return cv
