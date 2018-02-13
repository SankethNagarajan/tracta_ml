from setuptools import setup

setup(name='objective_ml',
      version='1.0',
      description='Taking subjectivity out of ML',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
      ],
      url='https://github.com/SankethNagarajan/objective_ml',
      author='Sanketh Nagarajan',
      author_email='sanketh.objml@gmail.com',
      license='BSD 3-Clause "New" or "Revised" License',
      packages=['objective_ml'],
      install_requires=['sklearn', 'numpy', 'pandas', 'pickle',\
                        'datetime','matplotlib','random','multiprocessing',
                        'joblib',
                        ],
      zip_safe=False)