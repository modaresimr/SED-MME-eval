
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='SED-MMR-Eval',
    version='1',
    description='Sound Event Recognition Multi-Modal Evaluation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='modaresi mr',
    author_email='modaresimr@gmail.com',
    url="https://github.com/modaresimr/SED-MME-eval",
    license='Apache Software License',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    namespace_packages=[],

    zip_safe=False,
    install_requires=[
        'numpy',
        'pandas',
        'wget',
        'ipympl',
        'intervaltree',
        'dcase_util',
        'scipy',
        'psds_eval',
        'matplotlib',
    ]
)
