import os
import platform
import shutil
import sys
import warnings
from os import path as osp

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


version_file = 'vvdet3d/version.py'


def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=None,
                  extra_args=None,
                  extra_include_path=None):
    if sources_cuda is None:
        sources_cuda = []
    if extra_args is None:
        extra_args = []
    if extra_include_path is None:
        extra_include_path = []

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    import re
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                yield ''.join(parts)

    return list(gen_packages_items())


def add_mim_extension():
    """Add files required by OpenMIM to the installed package."""
    if 'develop' in sys.argv:
        mode = 'copy' if platform.system() == 'Windows' else 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs', 'demo']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'vvdet3d', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        src_path = osp.join(repo_path, filename)
        tar_path = osp.join(mim_path, filename)

        if osp.isfile(tar_path) or osp.islink(tar_path):
            os.remove(tar_path)
        elif osp.isdir(tar_path):
            shutil.rmtree(tar_path)

        if mode == 'symlink':
            src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
            os.symlink(src_relpath, tar_path)
        elif mode == 'copy':
            if osp.isfile(src_path):
                shutil.copyfile(src_path, tar_path)
            elif osp.isdir(src_path):
                shutil.copytree(src_path, tar_path)
            else:
                warnings.warn(f'Cannot copy file {src_path}.')
        else:
            raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='vvdet3d',
        version=get_version(),
        description='Collaborative 3D perception framework for VVSim.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='VVDetection3D Contributors',
        keywords='computer vision, collaborative perception, 3d detection',
        url='https://github.com/LOTEAT/vvdetection3d',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'mim': parse_requirements('requirements/mminstall.txt'),
        },
        ext_modules=[],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
