import io
import os
import re
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command
import importlib.util

# Package arguments
spec = importlib.util.spec_from_file_location('package_info', 'warprnnt_numba/package_info.py')
package_info = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_info)

PACKAGE_NAME = package_info.__package_name__
DIRECTORY_NAME = package_info.__package_name__
SHORT_DESCRIPION = package_info.__description__
URL = package_info.__repository_url__
LICENCE = package_info.__license__

# Extra requirements and configs
EXTRA_REQUIREMENTS = {
    'cpu': ['pytorch'],
    'gpu': ['pytorch'],
}

# Test requirements and configs
TEST_REQUIRES = ['pytest']

REQUIRED_PYTHON = ">=3.7.0"  # Can be None, or a string value

# Signature arguments
AUTHOR = package_info.__contact_names__
EMAIL = package_info.__contact_emails__


###############################################################

# Attach test requirements to `tests`
EXTRA_REQUIREMENTS['tests'] = TEST_REQUIRES

base_path = os.path.abspath(os.path.dirname(__file__))

if LICENCE is None or LICENCE == '':
    raise RuntimeError("Licence must be provided !")

if os.path.exists(os.path.join(base_path, 'LICENCE')):
    raise RuntimeError("Licence must be provided !")


def get_version():
    """Return package version as listed in `__version__` in `init.py`."""
    return package_info.__version__


try:
    with open(os.path.join(base_path, 'requirements.txt'), encoding='utf-8') as f:
        REQUIREMENTS = f.read().split('\n')

except Exception:
    REQUIREMENTS = []

try:
    with io.open(os.path.join(base_path, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = '\n' + f.read()

except FileNotFoundError:
    LONG_DESCRIPTION = SHORT_DESCRIPION


class UploadCommand(Command):
    description = 'Build, install and upload tag to git with cleanup.'
    user_options = []

    def run(self):
        try:
            self.status('Removing previous builds...')
            rmtree(os.path.join(base_path, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel'.format(sys.executable))

        self.status('Pushing git tags...')
        os.system('git tag v{0}'.format(get_version()))
        os.system('git push --tags')

        try:
            self.status('Removing build artifacts...')
            rmtree(os.path.join(base_path, 'build'))
            rmtree(os.path.join(base_path, '{}.egg-info'.format(PACKAGE_NAME)))
        except OSError:
            pass

        sys.exit()

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    @staticmethod
    def status(s):
        print(s)


setup(
    name=PACKAGE_NAME,
    version=get_version(),
    packages=find_packages(exclude=['tests', 'scripts']),
    url=URL,
    download_url=URL,
    python_requires=REQUIRED_PYTHON,
    license=LICENCE,
    author=AUTHOR,
    author_email=EMAIL,
    description=SHORT_DESCRIPION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    test_suite="tests",
    tests_require=TEST_REQUIRES,
    # python setup.py upload
    cmdclass={
        'upload': UploadCommand,
    },
)
