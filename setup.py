from distutils.core import setup

setup(
    name='raha',
    packages=['tools.dBoost.dboost', 'tools.dBoost.dboost.utils', 'tools.dBoost.dboost.models',
              'tools.dBoost.dboost.features', 'tools.dBoost.dboost.analyzers', 'tools.dBoost.graphics.utils'],
    requires=['pandas', 'numpy', 'scipy', 'sklearn', 'IPython']
)
