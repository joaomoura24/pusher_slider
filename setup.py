from setuptools import setup, find_packages

# Install sliding_pack:
# 1. $ cd /path/to/push_slider
# 2. $ pip3 install .

setup(
    name='sliding_pack',
    version='1.0',
    url='https://github.com/joaomoura24/pusher_slider',
    author='Joao Moura',
    license='Apache v2.0',
    packages=find_packages(),
    package_data={'': ['*.yaml']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        # also requires CasADi, this is left out to prevent conflicts
        # with local installations from source
    ],
    zip_safe=False
)
