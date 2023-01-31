#%%
from src import construct
import pytest

def test_construct_cv_skf():
    nfolds = 10
    random_state = None
    shuffle = False
    cvtype = 'skf'
    cv = construct.construct_cv(cvtype=cvtype, nfolds=nfolds, random_state=random_state, shuffle = shuffle)

    assert cv.n_splits == nfolds
    assert cv.random_state == random_state
    assert cv.shuffle == shuffle
    assert cv.__class__.__name__ == 'StratifiedKFold'

def test_construct_cv_kf():
    nfolds = 10
    random_state = None
    shuffle = False
    cvtype = 'kf'
    cv = construct.construct_cv(cvtype=cvtype, nfolds=nfolds, random_state=random_state, shuffle = shuffle)

    assert cv.n_splits == nfolds
    assert cv.random_state == random_state
    assert cv.shuffle == shuffle
    assert cv.__class__.__name__ == 'KFold'

def test_construct_cv_error():
    nfolds = 10
    random_state = None
    shuffle = False
    cvtype = 'error'
    with pytest.raises(ValueError):
        cv = construct.construct_cv(cvtype=cvtype, nfolds=nfolds, random_state=random_state, shuffle = shuffle)

# %%
