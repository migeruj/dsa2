

pwd
which python
ls .

# python example/classifier/outlier_predict.py  preprocess  ;
# python example/classifier/outlier_predict.py  train    --nsample 1000     ;
# python example/classifier/outlier_predict.py  predict  --nsample 1000   ;

# python example/classifier/classifier_multi.py  train    --nsample 10000   ;


python example/regress/regress_salary.py  train   --nsample 1000
python example/regress/regress_salary.py  predict  --nsample 1000

python example/regress/regress_cardif.py  train   --nsample 1000


python example/regress/regress_airbnb.py  train   --nsample 20000
python example/regress/regress_airbnb.py  predict  --nsample 5000



python example/classifier/classifier_income.py  train    --nsample 1000   ;
python example/classifier/classifier_income.py  predict  --nsample 1000   ;


### HyperOpt optuna testing
python  example/test_hyperopt.py  hyperparam  --ntrials 1

### Keras model test
python example/test_mkeras.py  train    --config config1
