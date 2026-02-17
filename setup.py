from setuptools import setup, find_packages

# run pipdeptree --freeze --warn silence | grep -v -P "(^\-e|@\ file)" > requirements.txt
# to generate requirements.txt

with open('requirements.txt') as f:
  lines = f.read().splitlines()
  lines = [line.strip() for line in lines]
  lines = [line for line in lines if line and not line.startswith('#')]
  install_requires = lines

setup(
  name='unakinml',
  version='0.0.1',
  description='Unakin Machine Learning',
  author='Unakin',
  packages=find_packages(),
  # install_requires=install_requires,
)