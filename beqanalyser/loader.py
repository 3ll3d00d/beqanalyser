import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass

import numpy as np
import requests
from scipy.signal import unit_impulse, sosfilt, freqz

from beqanalyser import CatalogueEntry, ComplexFilter, BEQFilter

logger = logging.getLogger()


def convert(entry: CatalogueEntry, fs=1000) -> BEQFilter | None:
    u_i = unit_impulse(fs * 4, 'mid') * 23453.66
    f = ComplexFilter(fs=fs, filters=entry.iir_filters(fs=fs), description=f'{entry.digest}')
    try:
        filtered = sosfilt(f.get_sos(), u_i)
        w, h = freqz(filtered, worN=1 << (int(fs / 2) - 1).bit_length())
        x = w * fs * 1.0 / (2 * np.pi)
        h[h == 0] = 0.000000001
        return BEQFilter(x, 20 * np.log10(abs(h)), entry)
    except Exception as e:
        logger.exception(f'Unable to process entry {entry.title}')
        return None


class CatalogueEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, CatalogueEntry):
            return obj.for_search
        return super().default(obj)


class CatalogueDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(dct):
        if 'mag_freqs' in dct and 'mag_db' in dct and 'entry' in dct:
            return BEQFilter(np.array(dct['mag_freqs']), np.array(dct['mag_db']), CatalogueEntry('0', dct['entry']))
        return dct


def load() -> list[BEQFilter]:
    a = time.time()

    try:
        with open('database.bin', 'r') as f:
            data: list[dict] = json.load(f, cls=CatalogueDecoder)['data']
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.exception(f'Unable to load catalogue from database.bin, trying github')
        try:
            logger.info('Loading catalogue from github')
            r = requests.get('https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json',
                             allow_redirects=True)
            r.raise_for_status()
            with ProcessPoolExecutor() as executor:
                data: list[BEQFilter] = list(executor.map(convert, (CatalogueEntry(f'{idx}', e) for idx, e in enumerate(json.loads(r.content)) if e.get('filters', []))))
            with open('database.bin', 'w') as f:
                json.dump({'data': data}, f, cls=CatalogueEncoder)
        except requests.exceptions.HTTPError as e:
            logger.exception(f'Unable to load catalogue from database')
            raise e

    b = time.time()
    logger.info(f'Loaded catalogue in {b - a:.3g}s')

    return data
