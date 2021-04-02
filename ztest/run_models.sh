#!/bin/bash

set -e

pwd
which python
ls .
cd source/models


python torch_tabular.py test


python model_vaemdn.py test


python model_bayesian_pyro.py test


# python keras_widedeep.py test 


# python model_sampler.py test



