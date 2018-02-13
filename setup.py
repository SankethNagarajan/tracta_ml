from setuptools import setup

setup(name='tracta_ml',
      version='1.0',
      description='Taking subjectivity out of ML',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
      ],
      url='https://github.com/SankethNagarajan/tracta_ml',
      author='Sanketh Nagarajan',
      author_email='sanketh.objml@gmail.com',
      license='BSD 3-Clause "New" or "Revised" License',
      packages=['tracta_ml'],
      install_requires=['sklearn', 'numpy', 'pandas',\
                        'datetime','matplotlib',
                        ],
      zip_safe=False)