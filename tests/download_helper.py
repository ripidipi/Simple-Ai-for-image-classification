import requests
from pathlib import Path


if Path('helper_function.py').is_file():
    print('help exist')
else:
    print('download help')
    request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('downloads//helper_functions.py', 'wb') as f:
        f.write(request.content)
