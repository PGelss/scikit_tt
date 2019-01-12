from setuptools import setup, find_packages

metadata = dict(name='scikit_tt',
                author='Patrick Gelss',
                author_email='p.gelss@fu-berlin.de',
                license='LGPLv3',
                version='1.0',
                packages=find_packages(),
                )

if __name__ == '__main__':
    setup(**metadata)
