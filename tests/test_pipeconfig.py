#%%
import os
import json
from src import pipeconfig
from src.transform import SelectCols
from pathlib import Path
from sklearn.linear_model import BayesianRidge, Ridge

TEST_FILE = Path(__file__).parent / 'test_files' / 'test_pipeconfig.json'

def test_get_spec():
    test_spec = {'one':{'two':2}}
    spec = pipeconfig.get_spec(test_spec, 'one')
    assert spec['two'] == 2

def test_get_pipeline_config_prproc():
    pipe_spec = pipeconfig.get_pipeline_config(TEST_FILE)
    preproc = pipeconfig.get_spec(pipe_spec,'preproc')
    assert preproc['steps']['step1']['object'] == 'SelectCols'
    assert preproc['steps']['step1']['kwargs']['keep'] == '[1]'
    assert preproc['steps']['step2']['object'] == 'SelectCols'
    assert preproc['steps']['step2']['kwargs']['keep'] == '[2]'

def test_get_pipeline_config_estimator():
    pipe_spec = pipeconfig.get_pipeline_config(TEST_FILE)
    est1 = pipeconfig.get_spec(pipe_spec,'est1')
    est2 = pipeconfig.get_spec(pipe_spec,'est2')
    assert est1['steps']['model']['object'] == 'BayesianRidge'
    assert est2['steps']['model']['object'] == 'Ridge'

def test_get_cv_config():
    cv = pipeconfig.get_cv_config(TEST_FILE)
    assert cv['cv_inner']['object'] == 'KFold'
    assert cv['cv_inner']['kwargs']['n_splits'] == 5
    assert cv['cv_inner']['kwargs']['shuffle'] == True
    assert cv['cv_inner']['kwargs']['random_state'] == 42
    assert cv['cv_outer']['object'] == 'StratifiedKFold'
    assert cv['cv_outer']['kwargs']['n_splits'] == 5
    assert cv['cv_outer']['kwargs']['shuffle'] == True
    assert cv['cv_outer']['kwargs']['random_state'] == 42

def test_get_paramgrid_config():
    param_grid = pipeconfig.get_paramgrid_config(TEST_FILE)
    assert param_grid['est1']['param1'] == [1,2,3]
    assert param_grid['est2']['param2'] == [4,5,6]

def test_make_eval_dict():
    steps = ['stp1','stp2']
    kwargs = ['v1','v2']
    values = [['1','2'],'3']
    pass_params = pipeconfig.make_eval_dict(steps, kwargs, values)
    assert pass_params['stp1']['kwarg'] == 'v1'
    assert pass_params['stp1']['value'] == ['1','2']
    assert pass_params['stp2']['kwarg'] == 'v2'
    assert pass_params['stp2']['value'] == '3'

def test_get_steps_from_spec():
    steps = ['step1','step2']
    kwargs = ['keep','keep']
    values = [[1],[2]]
    eval_dict = pipeconfig.make_eval_dict(steps, kwargs, values)
    pipe_spec = pipeconfig.get_pipeline_config(TEST_FILE)
    preproc = pipeconfig.get_spec(pipe_spec,'preproc')
    steps = pipeconfig.get_steps_from_spec(preproc, eval_dict=eval_dict)
    assert steps[0][0] == 'step1'
    assert steps[0][1].keep == [1]
    assert steps[1][0] == 'step2'
    assert steps[1][1].keep == [2]

def test_get_kwargs():
    pipe = pipeconfig.get_pipeline_config(TEST_FILE)
    preproc_spec = pipeconfig.get_spec(pipe, 'preproc')
    kwargs1 = pipeconfig.get_kwargs(preproc_spec, 'step1')
    kwargs2 = pipeconfig.get_kwargs(preproc_spec, 'step2')
    assert kwargs1['keep'] == '[1]'
    assert kwargs2['keep'] == '[2]'

def test_get_object_name():
    pipe = pipeconfig.get_pipeline_config(TEST_FILE)
    preproc_spec = pipeconfig.get_spec(pipe, 'preproc')
    obj_name1 = pipeconfig.get_object_name(preproc_spec, 'step1')
    obj_name2 = pipeconfig.get_object_name(preproc_spec, 'step2')
    assert obj_name1 == 'SelectCols'
    assert obj_name2 == 'SelectCols'

def test_eval_kwargs():
    kwargs = {'columns':'cols'}
    kwargs = pipeconfig.eval_kwargs(kwargs, eval_dict={'kwarg':'columns', 'value':[1]})
    assert kwargs['columns'] == [1]

def test_get_object_from_name():
    kwargs = {'keep':[1]}
    obj = pipeconfig.get_object_from_name('SelectCols', **kwargs)
    assert isinstance(obj, SelectCols)
    assert obj.keep == [1]

def test_build_pipeline():
    steps = ['step1','step2']
    kwargs = ['keep','keep']
    values = [[1],[2]]
    eval_dict = pipeconfig.make_eval_dict(steps, kwargs, values)
    pipe = pipeconfig.build_pipeline(TEST_FILE,custom='est1', eval_dict=eval_dict)
    assert isinstance(pipe.steps[0][1], SelectCols)
    assert pipe.steps[0][1].keep == [1]
    assert isinstance(pipe.steps[1][1], SelectCols)
    assert pipe.steps[1][1].keep == [2]
    assert isinstance(pipe.steps[2][1], BayesianRidge)

def test_build_cross_val():
    cv_inner, cv_outer = pipeconfig.build_cross_val(TEST_FILE)
    assert cv_inner.n_splits == 5
    assert cv_inner.shuffle == True
    assert cv_inner.random_state == 42
    assert cv_outer.n_splits == 5
    assert cv_outer.shuffle == True
    assert cv_outer.random_state == 42

def test_get_param_grid():
    pipe_name_1 = 'est1'
    pipe_name_2 = 'est2'
    param_grid_1 = pipeconfig.get_param_grid(TEST_FILE,pipe_name_1)
    param_grid_2 = pipeconfig.get_param_grid(TEST_FILE,pipe_name_2)
    assert param_grid_1['param1'] == [1,2,3]
    assert param_grid_2['param2'] == [4,5,6]

# %%
