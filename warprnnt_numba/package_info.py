MAJOR = 0
MINOR = 4
PATCH = 0
PRE_RELEASE = ''

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__short_version__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'warprnnt_numba'
__contact_names__ = 'Somshubra Majumdar'
__contact_emails__ = 'titu1994@gmail.com'
__homepage__ = 'https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/'
__repository_url__ = 'https://github.com/titu1994/warprnnt_numba'
__download_url__ = 'https://github.com/titu1994/warprnnt_numba/releases'
__description__ = 'Warp RNNT loss ported to Numba for faster experimentation'
__license__ = 'MIT'
