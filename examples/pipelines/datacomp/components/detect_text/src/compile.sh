export MKL_SERVICE_FORCE_INTEL=1

cd ./models/post_processing/pa/
python setup.py build_ext --inplace
cd ../pse/
python setup.py build_ext --inplace
cd ../ccl/
python setup.py build_ext --inplace
cd ../../../