#!/usr/bin/env python

from setuptools import setup

def main():
    setup(
        name="pyneural",
        version="0.1",
        description="Neural Network Library for Python",
        long_description="""
        PyNeural provides a probabilistic random weight change algorithm implemented in python.
        It should eventually take advantage of GPU's via python-cuda/pycuda.  
        """,
        author=u"Justin Riley",
        author_email="justin.t.riley@gmail.com",
        license = "GPL2",
        url="http://web.mit.edu/jtriley/www",
        classifiers=[
          'Environment :: Console',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          ],

        # build info
        packages=['pyneural'],

        install_requires=[
            "numpy",
        ],

        package_dir={"pyneural": "pyneural"},
        zip_safe = True,
    )

if __name__ == '__main__':
    main()
