from pathlib import Path

import sllib

from models import ExtendedFrame

SUPPORTED_TYPES = ('slg', 'sl2', 'sl3')


def _load_file(path: str) -> [ExtendedFrame]:
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} does not point to an existing file')

    _, ending = path.lower().rsplit('.', 1)

    if ending not in SUPPORTED_TYPES:
        raise NotImplementedError(
            '{} is not a supported file-type, only {} is supported'.format(ending, ', '.join(SUPPORTED_TYPES))
        )

    with open(path, 'rb') as f:
        reader = sllib.Reader(f)
        return list(map(ExtendedFrame, reader))


def load_files(path: str | list[str]) -> dict[str, list[ExtendedFrame]]:
    path_obj = Path(path)

    if path_obj.is_dir():
        file_splits = [list(path_obj.glob(f'*.{ext}')) for ext in SUPPORTED_TYPES]
        path = []

        for fs in file_splits:
            path.extend([str(f.absolute()) for f in fs])

    if type(path) == str:
        path = [path]

    if type(path) != list:
        raise AttributeError('Only list is supported')

    return {Path(p).name: _load_file(p) for p in path}
