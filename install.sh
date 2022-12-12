export DISCOPATH=${PWD}
echo $DISCOPATH

'''
pip install --upgrade git+https://github.com/BorealisAI/advertorch.git
pip install --upgrade git+https://github.com/fra31/auto-attack.git@6482e4d6fbeeb51ae9585c41b16d50d14576aadc#egg=autoattack
cd testing
python setup.py install
cd ../
'''
