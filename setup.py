from setuptools import setup, find_packages
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name='de-tabsync',
    version='0.0.1',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/DoraDong-2023/DecomposableTabSyn',
    install_requires=requirements,
)
