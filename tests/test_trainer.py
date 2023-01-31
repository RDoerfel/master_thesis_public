#%%
from src import trainer
from src import construct
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
import logging

#%%

def _generate_data():
    """ Generate zeros for testing """
    # simulate data
    data = np.zeros((10, 3))
    df = pd.DataFrame(data, columns=['cat1', 'pyment', 'age'])
    return df


def test_logger_default():
    expected_logger = logging.getLogger()
    expected_logger.addHandler(logging.StreamHandler())
    expected_logger.setLevel(logging.INFO)
    train = trainer.Trainer()
    assert train.logger == expected_logger

def test_logger_custom():
    expected_logger = logging.getLogger()
    expected_logger.addHandler(logging.StreamHandler())
    expected_logger.setLevel(logging.DEBUG)
    train = trainer.Trainer(logger=expected_logger)
    assert train.logger == expected_logger

def test_trainer_scores():
    data = _generate_data()

    # bin data
    max_val = 10.0
    min_val = 0.0
    step = 2
    col = 'age'
    data = trainer.bin_data(data, col, min_val=min_val, max_val=max_val, step=step)

    # construct cv
    cv_outer = construct.construct_cv(cvtype='kf', nfolds=2, random_state=42, shuffle = True)
    cv_inner = construct.construct_cv(cvtype='kf', nfolds=2, random_state=42, shuffle = True)

    # create dummy pipeline
    pipe = Pipeline(steps=[('model', DummyRegressor())])

    train = trainer.Trainer(logger=None)
    train.train(data=data,
                strat_col='age_group',
                label_col='age',
                pipeline=pipe,
                cv_inner=cv_inner,
                cv_outer=cv_outer,
                param_grid={})

    scores = train.get_scores()
    assert np.all(scores['r2'] == np.array([1.0, 1.0]))
    assert np.all(scores['r2_pyment'] == np.array([1.0, 1.0]))
    assert np.all(scores['mae'] == np.array([0.0, 0.0]))
    assert np.all(scores['mae_pyment'] == np.array([0.0, 0.0]))

def test_trainer_predictions():
    data = _generate_data()

    # bin data
    max_val = 10.0
    min_val = 0.0
    step = 2
    col = 'age'
    data = trainer.bin_data(data, col, min_val=min_val, max_val=max_val, step=step)

    # construct cv
    cv_outer = construct.construct_cv(cvtype='kf', nfolds=2, random_state=42, shuffle = True)
    cv_inner = construct.construct_cv(cvtype='kf', nfolds=2, random_state=42, shuffle = True)

    # create dummy pipeline
    pipe = Pipeline(steps=[('model', DummyRegressor())])

    train = trainer.Trainer(logger=None)
    train.train(data=data,
                strat_col='age_group',
                label_col='age',
                pipeline=pipe,
                cv_inner=cv_inner,
                cv_outer=cv_outer,
                param_grid={})

    results = train.get_results()

    assert np.all(results['pred'] == 0.0)
    assert np.all(results['std']== 0.0)
    assert np.all(results['true'] == 0.0)
    assert np.all(results['pyment'] == 0.0)
    
    
if __name__ == '__main__':
    test_logger_default()
    test_logger_custom()
    test_trainer_scores()
    test_trainer_predictions()


# %%
