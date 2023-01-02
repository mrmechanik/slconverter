import logging
from pathlib import Path

import sllib

from models import ExtractedFrame

logger = logging.getLogger(__name__)
supported_types = ('slg', 'sl2', 'sl3')


def load_file(path: str) -> list[ExtractedFrame]:
    if not Path(path).exists():
        logger.debug(f'{path} does not exist')
        raise FileNotFoundError(f'{path} does not point to an existing file')

    _, ending = path.lower().rsplit('.', 1)

    if ending not in supported_types:
        logger.debug('Unsupported file detected: {}, is not one of {}'.format(path, ', '.join(supported_types)))
        raise NotImplementedError(
            '{} is not a supported file-type, only {} is supported'.format(ending, ', '.join(supported_types))
        )

    logger.debug(f'Reading {path} ...')

    with open(path, 'rb') as f:
        reader = sllib.Reader(f)
        frames = list(map(ExtractedFrame, reader))
        logger.debug(f'{len(frames)} frames found and mapped')
        return frames


def load_files(path: str | list[str]) -> dict[str, list[ExtractedFrame]]:
    path_obj = Path(path)

    if path_obj.is_dir():
        logger.debug(f'{path} is a directory, now searching readable files in it')
        file_splits: list[list[Path]] = [list(path_obj.glob(f'*.{ext}')) for ext in supported_types]
        path: list[str] = [str(f.absolute()) for sub_fs in file_splits for f in sub_fs]
        logger.debug(f'Directory contains {len(path)} readable files')

    if type(path) == str:
        logger.debug(f'Singe file input detected ({path})')
        path: list[str] = [path]

    if type(path) != list:
        logger.debug(f'{path} is not a list')
        raise AttributeError('Only list is supported')

    logger.debug(f'Now processing {len(path)} files...')
    return {Path(p).name: load_file(p) for p in path}
