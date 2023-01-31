#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
import sys
from pathlib import Path

from src import json_handling
from src.transform import DiMap, SelectCols, OneHotPD
from src.estimator import GPR

def get_spec(pipe_spec,spec):
    """Get spec from pipe_spec
    Args:
        pipe_spec (dict): pipeline
        spec (str): spec name
    Returns:
        spec (dict): spec"""
    return pipe_spec.get(spec)

def get_config(pipe_config_file, pipe_step):
    """Get configuration for pipe_step from configurtation file
    Args:
        pipe_config_file (string): pipeline
        pipe_step (str): part of pipeline, e.g. 'covariates', 'pipeline', 'cross-validation' or 'param_grid'
    Returns:
        pipe_spec (dict): pipeline specification"""
    pipe_spec = json_handling.read_json(pipe_config_file)
    validate_config(pipe_spec)
    return get_spec(pipe_spec,pipe_step)

def get_covarities(pipe_config_file):
    """Get covariates from configurtation file
    Args:
        pipe_config_file (string): pipeline
    Returns:
        covariates (dict): covariates"""
    # get pipeline configuration
    return get_config(pipe_config_file, 'covariates')

def get_pipeline_config(pipe_config_file):
    """Get pipeline configuration from configurtation file
    Args:
        pipe_config_file (string): pipeline configuration file
        pipe_name (str): name of pipeline
    Returns:
        pipe_config (dict): pipeline configuration"""
    # get pipeline configuration
    return get_config(pipe_config_file, 'pipeline')

def get_paramgrid_config(pipe_config_file):
    """Get paramgrid configuration from configurtation file
    Returns:
        paramgrid_config (dict): paramgrid configuration"""
    # get pipeline configuration
    return get_config(pipe_config_file, 'param_grid')

def get_cv_config(pipe_config_file):
    """Get cross validation configuration from pipe_config
    Args:
        pipe_config_file (string): pipeline
        pipe_name (str): name of pipeline
    Returns:
        cv_config (dict): cross validation configuration"""
    # get pipeline configuration
    return get_config(pipe_config_file, 'cross-validation')

def get_object_from_name(obj_name, **kwargs):
    """Get object from class name and kwargs"""
    # get class from string
    class_ = getattr(sys.modules[__name__], obj_name)
    # instantiate class
    instance = class_(**kwargs)
    return instance

def eval_kwargs(kwargs, eval_dict):
    """Evaluate kwargs in eval_dict
    Args:
        kwargs (dict): kwargs to evaluate
        eval_dict (dict): dict of kwarg and value
    Returns:
        kwargs (dict): evaluated kwargs"""
    # evaluate kwargs in eval_dict
    kwarg = eval_dict['kwarg']
    kwargs[kwarg] = eval_dict['value']

    return kwargs

def get_object_name(pipe_spec,step):
    """Get object name from step
    Args:
        pipe_spec (dict): pipeline specification
        step (str): step name
    Returns:    
        obj_name (str): object name"""
        # get object name
    return get_spec(pipe_spec['steps'][step],'object')

def get_kwargs(pipe_spec, step):
    """Get kwargs from step
    Args:
        pipe_spec (dict): pipeline
        step (str): step name
    Returns:
        kwargs (dict): kwargs"""
    # get kwargs
    return get_spec(pipe_spec['steps'][step],'kwargs')

def get_steps_from_spec(pipe_spec,eval_dict=None):
    """Get steps from pipe_spec
    Args:
        pipe_spec (dict): pipeline specification
        eval_dict (dict): list of kwargs to evaluate
    Returns:
        steps (list): list of steps"""
    steps = []
    for step in pipe_spec['steps']:
        
        obj_name = get_object_name(pipe_spec, step)
        kwargs = get_kwargs(pipe_spec, step)
        if step in eval_dict:
            kwargs = eval_kwargs(kwargs, eval_dict[step])
                
        # get object
        obj = get_object_from_name(obj_name, **kwargs)

        steps.append((step,obj))
    return steps

def build_pipeline(pipe_config_file, custom, eval_dict=None):
    """build pipeline from pipe_spec
    Args:
        pipe_spec (dict): pipeline
        custom (list): name of custom steps to include
        eval_dict (dict): list of kwargs to evaluate
    Returns:
        pipe (Pipeline): pipeline"""
        
    # get preprocessing steps
    pipe_spec = get_pipeline_config(pipe_config_file)
    preproc_specs = get_spec(pipe_spec,'preproc')
    preproc_steps = get_steps_from_spec(preproc_specs, eval_dict)

    # get custom steps
    custom_specs = get_spec(pipe_spec, custom)
    custom_steps = get_steps_from_spec(custom_specs, eval_dict)

    # combine steps
    steps = preproc_steps + custom_steps

    # make pipeline
    pipe = Pipeline(steps=steps)
    return pipe

def build_cross_val(pipe_config_file):
    """build cross validations from pipe_spec 
    Args:
        pipe_config_file (string): pipeline configuration file
    Returns:    
        cv_inner (sklearn.model_selection): inner cross validation
        cv_outer (sklearn.model_selection): outer cross validation"""
    cv_spec = get_cv_config(pipe_config_file)
    cv_inner = get_object_from_name(cv_spec['cv_inner']['object'], **cv_spec['cv_inner']['kwargs'])
    cv_outer = get_object_from_name(cv_spec['cv_outer']['object'], **cv_spec['cv_outer']['kwargs'])
    return cv_inner, cv_outer

def get_param_grid(pipe_config_file, pipe_name):
    """Get param grid from pipe_spec
    Args:
        pipe_config_file (str): pipeline configuration file
        pipe_name (str): name of pipeline
    Returns:
        param_grid (dict): param grid"""
    param_config = get_paramgrid_config(pipe_config_file)
    return get_spec(param_config,pipe_name)

def make_eval_dict(steps,kwargs,values):
    """Make pass_param_dict for pipeline
    Args:
        steps (list): list of steps
        kwargs (list): list of kwargs
        values (list): list of values
    Returns:
        pass_param_dict (dict): pass_param_dict"""
    pass_param_dict = {}
    for step, kwarg, value in zip(steps,kwargs,values):
        pass_param_dict[step]={'kwarg':kwarg,'value':value}
    return pass_param_dict

    
def validate_spec(spec,list_of_keys):
    """Validate spec, fails if something is missing
    Args:
        spec (dict): spec
        list_of_keys (list): list of keys
    Returns:
        valid (bool): whether spec is valid or not"""
    valid = True
    for key in list_of_keys:
        if key not in spec:
            valid = False
            print('missing key: {}'.format(key))
            return valid
    return valid

def validate_config_cv(cv_spec):
    """ Validate cv_spec, fails if something is missing
    Args:
        cv_spec (dict): cv_spec
    Returns:
        valid (bool): whether cv_spec is valid or not"""
    valid = True
    # check if cv_inner and cv_outer are in cross-validation
    assert validate_spec(cv_spec,['cv_inner','cv_outer'])
    cv_inner_spec = get_spec(cv_spec,'cv_inner')
    cv_outer_spec = get_spec(cv_spec,'cv_outer')
    assert validate_spec(cv_inner_spec,['object','kwargs'])
    assert validate_spec(cv_outer_spec,['object','kwargs'])
    return valid

def validate_config_pipe(pipe_spec):
    """Validate pipe_spec, fails if something is missing
    Args:
        pipe_spec (dict): pipe_spec
    Returns:
        valid (bool): whether pipe_spec is valid or not"""
    valid = True
    # check if preproc is in pipeline
    assert validate_spec(pipe_spec,['preproc'])
    # check if all parts of the pipeline have necessary structure
    list_pipe_keys = pipe_spec.keys()
    for pipe_key in list_pipe_keys:
        pipe = get_spec(pipe_spec,pipe_key)
        # check for steps
        assert validate_spec(pipe,['steps'])
        for step_key in pipe['steps']:
            step_spec = get_spec(pipe['steps'],step_key)
            # check for object and kwargs
            assert validate_spec(step_spec,['object','kwargs'])
    return valid

def validate_config(pipeline_specs):
    """Validate config, fails if something is missing
    Args:
        pipeline_specs (dict): path to pipe_file
        custom (str): str of custom steps"""
    pipe_spec = get_spec(pipeline_specs,'pipeline')
    cv_spec = get_spec(pipeline_specs,'cross-validation')
    param_grid = get_spec(pipeline_specs,'param_grid')
    assert validate_config_cv(cv_spec)
    assert validate_config_pipe(pipe_spec)
    assert param_grid is not None

# %%
