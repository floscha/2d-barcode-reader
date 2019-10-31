from setuptools import setup


setup(
    name='barcode_reader',
    version='0.0.1',
    description='A Python/OpenCV-based barcode reader for 2D barcode markers.',
    url='https://github.com/floscha/2d-barcode-reader',
    author='Florian Schäfer',
    author_email='florian.joh.schaefer@gmail.com',
    license='MIT',
    packages=['barcode_reader'],
    install_requires=[
        'numpy==1.17.3',
        'opencv-python==3.4.7.28'
    ],
    zip_safe=False
)
