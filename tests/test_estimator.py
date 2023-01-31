#%%
from src import estimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct
import numpy as np

def test_estimator_gpr_kernel_none():
    gpr_expected = GaussianProcessRegressor()
    gpr = estimator.GPR(kernel=None)
    assert gpr.kernel == gpr_expected.kernel
    assert gpr.alpha == gpr_expected.alpha

def test_estimator_gpr_kernel_invalid():
    gpr_expected = GaussianProcessRegressor()
    gpr = estimator.GPR(kernel='invalid')
    assert gpr.kernel == gpr_expected.kernel
    assert gpr.alpha == gpr_expected.alpha

def test_estimator_gpr_kernel_rbf():
    kernel = RBF()
    gpr_expected = GaussianProcessRegressor(kernel=kernel)
    gpr = estimator.GPR(kernel='RBF')
    assert gpr.kernel == gpr_expected.kernel
    assert gpr.alpha == gpr_expected.alpha

def test_estimator_gpr_kernel_kwargs():
    kernel = RBF(length_scale=45.0)
    gpr_expected = GaussianProcessRegressor(kernel=kernel)
    gpr = estimator.GPR(kernel='RBF', kernel_wargs={'length_scale': 45.0})
    assert gpr.kernel == gpr_expected.kernel

def test_estimator_gpr_kernel_instance_rbf():
    kernel = RBF()
    gpr_expected = GaussianProcessRegressor(kernel=kernel)
    gpr = estimator.GPR(kernel=kernel)
    assert gpr.kernel == gpr_expected.kernel
    assert gpr.alpha == gpr_expected.alpha

def test_estimator_gpr_kernel_instance_kwargs():
    
    gpr_expected = GaussianProcessRegressor(kernel=RBF(length_scale=45.0))
    gpr = estimator.GPR(kernel=RBF(), kernel_wargs={'length_scale': 45.0})
    assert gpr.kernel == gpr_expected.kernel

def test_estimator_gpr_fit():
    # setup kernel and estimator
    kernel = "RBF"
    gpr = estimator.GPR(kernel=kernel)
    # setup data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1., 2., 3.])
    # fit
    gpr.fit(X, y)
    # predict
    y_pred = gpr.predict(X)
    # check
    assert np.allclose(y, y_pred)




#%%
if __name__ == '__main__':
    test_estimator_gpr_kernel_none()
    test_estimator_gpr_kernel_invalid()
    test_estimator_gpr_kernel_rbf()
    test_estimator_gpr_kernel_kwargs()
    test_estimator_gpr_fit()
    test_estimator_gpr_kernel_instance_rbf()
    test_estimator_gpr_kernel_instance_kwargs()
# %%
