import inspect
import pathlib

def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)

def configuration(parent_package='', top_path=None):
    from scipy._build_utils.boostinator import get_include_dir
    from _generate_pyx import _klass_mapper
    from numpy.distutils.misc_util import Configuration
    import numpy as np
    config = Configuration('boost', parent_package, top_path)

    DEFINES = [
        ('BOOST_MATH_DOMAIN_ERROR_POLICY', 'ignore_error'),  # return nan instead of throwing
        ('BOOST_MATH_PROMOTE_DOUBLE_POLICY', 'false'),
    ]
    INCLUDES = [
        'include/',
        'src/',
        str(get_include_dir()),
        np.get_include(),
    ]

    # generate the PXD and PYX wrappers
    src_dir = pathlib.Path(__file__).parent / 'src'
    for s in _klass_mapper.values():
        ext = config.add_extension(
            f'{s.scipy_name}_ufunc',
            sources=[f'{src_dir}/{s.scipy_name}_ufunc.cxx'],
            include_dirs=INCLUDES,
            define_macros=DEFINES,
            language='c++',
        )
        # Add c++11/14 support:
        ext._pre_build_hook = pre_build_hook

    config.add_subpackage('_stats')
    config.add_subpackage('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
