from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='teddr',
    version='0.0.1',
    author='Lucas H. McCabe',
    author_email='lmccabe@lmi.org',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url='https://github.com/lmiconsulting/teddr',
    license='LICENSE.txt',
    description='transportation event distance distribution reconstruction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
