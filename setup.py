
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='wdwdl',
    version='1.0.0',
    description='Workaround detection with deep learning in event log data',
    long_description=readme,
    author='Sven Weinzierl',
    author_email='sven.weinzierl@fau.de',
    url='https://github.com/fau-is/wdwdl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

