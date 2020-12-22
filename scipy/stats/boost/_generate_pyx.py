import pathlib
from shutil import copyfile
from typing import NamedTuple

class _KlassMap(NamedTuple):
    scipy_name: str
    ctor_args: tuple

# map boost stats classes to scipy class names and number of
# constructor arguments; b -> (s, #)
_klass_mapper = {
    'beta': _KlassMap('beta', ('a', 'b')),
    'binomial': _KlassMap('binom', ('n', 'p')),
    'negative_binomial': _KlassMap('nbinom', ('n', 'p')),
    'non_central_chi_squared': _KlassMap('ncx2', ('df', 'nc')),
}


if __name__ == '__main__':
    # create target directory
    (pathlib.Path(__file__).parent / 'src').mkdir(exist_ok=True, parents=True)
    src_dir = pathlib.Path(__file__).parent / 'src'

    # copy contents of include into directory to satisfy Cython
    # PXD include conditions
    inc_dir = pathlib.Path(__file__).parent / 'include'
    src = 'templated_pyufunc.pxd'
    copyfile(inc_dir / src, src_dir / src)

    # generate the PXD and PYX wrappers
    from include.gen_func_defs_pxd import gen_func_defs_pxd
    from include.code_gen import ufunc_gen
    gen_func_defs_pxd(f'{src_dir}/func_defs.pxd')
    for b, s in _klass_mapper.items():
        ufunc_gen(
            wrapper_prefix=s.scipy_name,
            types=['NPY_DOUBLE', 'NPY_FLOAT'],
            num_ctor_args=len(s.ctor_args),
            filename=f'{src_dir}/{s.scipy_name}_ufunc.pyx',
            boost_dist=f'{b}_distribution',
        )
