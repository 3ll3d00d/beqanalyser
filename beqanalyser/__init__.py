import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger()
TWO_WEEKS_AGO_SECONDS = 2 * 7 * 24 * 60 * 60


class CatalogueEntry:

    def __init__(self, idx: str, vals: dict):
        self.idx = idx
        self.title = vals.get('title', '')
        self.casefolded_title = self.title.casefold()
        y = 0
        try:
            y = int(vals.get('year', 0))
        except:
            logger.error(f"Invalid year {vals.get('year', 0)} in {self.title}")
        self.year = y
        self.audio_types = vals.get('audioTypes', [])
        self.content_type = vals.get('content_type', 'film')
        self.author = vals.get('author', '')
        self.beqc_url = vals.get('catalogue_url', vals.get('beqcUrl', ''))
        self.filters: list[dict] = vals.get('filters', [])
        self.images = vals.get('images', [])
        self.warning = vals.get('warning', [])
        self.season = vals.get('season', '')
        self.episodes = vals.get('episode', '')
        self.avs_url = vals.get('avs', '')
        self.sort_title = vals.get('sortTitle', '')
        self.edition = vals.get('edition', '')
        self.note = vals.get('note', '')
        self.language = vals.get('language', '')
        self.source = vals.get('source', '')
        self.overview = vals.get('overview', '')
        self.the_movie_db = vals.get('theMovieDB', '')
        self.rating = vals.get('rating', '')
        self.genres = vals.get('genres', [])
        self.altTitle = vals.get('altTitle', '')
        self.created_at = vals.get('created_at', 0)
        self.updated_at = vals.get('updated_at', 0)
        self.digest = vals.get('digest', '')
        self.collection = vals.get('collection', {})
        self.formatted_title = self.__format_title()
        now = time.time()
        if self.created_at >= (now - TWO_WEEKS_AGO_SECONDS):
            self.freshness = 'Fresh'
        elif self.updated_at >= (now - TWO_WEEKS_AGO_SECONDS):
            self.freshness = 'Updated'
        else:
            self.freshness = 'Stale'
        try:
            r = int(vals.get('runtime', 0))
        except:
            logger.error(f"Invalid runtime {vals.get('runtime', 0)} in {self.title}")
            r = 0
        self.runtime = r
        self.mv_adjust = 0.0
        if 'mv' in vals:
            v = vals['mv']
            try:
                self.mv_adjust = float(v)
            except:
                logger.error(f"Unknown mv_adjust value in {self.title} - {vals['mv']}")
                pass
        self.for_search = {
            'id': self.idx,
            'title': self.title,
            'year': self.year,
            'sortTitle': self.sort_title,
            'audioTypes': self.audio_types,
            'contentType': self.content_type,
            'author': self.author,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'freshness': self.freshness,
            'digest': self.digest,
            'formattedTitle': self.formatted_title
        }
        if self.beqc_url:
            self.for_search['beqcUrl'] = self.beqc_url
        if self.images:
            self.for_search['images'] = self.images
        if self.warning:
            self.for_search['warning'] = self.warning
        if self.season:
            self.for_search['season'] = self.season
        if self.episodes:
            self.for_search['episodes'] = self.episodes
        if self.mv_adjust:
            self.for_search['mvAdjust'] = self.mv_adjust
        if self.avs_url:
            self.for_search['avsUrl'] = self.avs_url
        if self.edition:
            self.for_search['edition'] = self.edition
        if self.note:
            self.for_search['note'] = self.note
        if self.language:
            self.for_search['language'] = self.language
        if self.source:
            self.for_search['source'] = self.source
        if self.overview:
            self.for_search['overview'] = self.overview
        if self.the_movie_db:
            self.for_search['theMovieDB'] = self.the_movie_db
        if self.rating:
            self.for_search['rating'] = self.rating
        if self.runtime:
            self.for_search['runtime'] = self.runtime
        if self.genres:
            self.for_search['genres'] = self.genres
        if self.altTitle:
            self.for_search['altTitle'] = self.altTitle
        if self.note:
            self.for_search['note'] = self.note
        if self.warning:
            self.for_search['warning'] = self.warning
        if self.collection and 'name' in self.collection:
            self.for_search['collection'] = self.collection['name']

    def matches(self, authors: list[str], years: list[int], audio_types: list[str], content_types: list[str]):
        if not authors or self.author in authors:
            if not years or self.year in years:
                if not audio_types or any(a_t in audio_types for a_t in self.audio_types):
                    return not content_types or self.content_type in content_types
        return False

    def __repr__(self):
        return f"[{self.content_type}] {self.title} / {self.audio_types} / {self.year}"

    @staticmethod
    def __format_episodes(formatted, working):
        val = ''
        if len(formatted) > 1:
            val += ', '
        if len(working) == 1:
            val += working[0]
        else:
            val += f"{working[0]}-{working[-1]}"
        return val

    def __format_tv_meta(self):
        season = f"S{self.season}" if self.season else ''
        episodes = self.episodes.split(',') if self.episodes else None
        if episodes:
            formatted = 'E'
            if len(episodes) > 1:
                working = []
                last_value = 0
                for ep in episodes:
                    if len(working) == 0:
                        working.append(ep)
                        last_value = int(ep)
                    else:
                        current = int(ep)
                        if last_value == current - 1:
                            working.append(ep)
                            last_value = current
                        else:
                            formatted += self.__format_episodes(formatted, working)
                            working = []
                if len(working) > 0:
                    formatted += self.__format_episodes(formatted, working)
            else:
                formatted += f"{self.episodes}"
            return f"{season}{formatted}"
        return season

    def __format_title(self) -> str:
        if self.content_type == 'TV':
            return f"{self.title} {self.__format_tv_meta()}"
        return self.title

    def iir_filters(self, fs=96000) -> list['BiquadWithQGain']:
        return [self.__convert(i, f, fs) for i, f in enumerate(self.filters)]

    @staticmethod
    def __convert(i: int, f: dict, fs: int) -> 'BiquadWithQGain':
        t = f['type']
        freq = f['freq']
        gain = f['gain']
        q = f['q']
        if t == 'PeakingEQ':
            return PeakingEQ(fs, freq, q, gain, f_id=i)
        elif t == 'LowShelf':
            return LowShelf(fs, freq, q, gain, f_id=i)
        elif t == 'HighShelf':
            return HighShelf(fs, freq, q, gain, f_id=i)
        else:
            raise ValueError(f"Unknown filt_type {t}")


# from http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

COMBINED = 'Combined'


class SOS(ABC):

    def __init__(self, f_id=-1, fs=48000):
        self.__id = f_id
        self.fs = fs

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @abstractmethod
    def get_sos(self) -> list[list[float]] | None:
        pass


class Biquad(SOS):
    def __init__(self, fs, f_id=-1):
        super().__init__(f_id=f_id, fs=fs)
        self.a, self.b = self._compute_coeffs()
        self.__transfer_function = None

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.fs == o.fs
        return equal

    def __repr__(self):
        return self.description

    @property
    def description(self):
        description = ''
        if hasattr(self, 'display_name'):
            description += self.display_name
        return description

    @property
    def filter_type(self):
        return self.__class__.__name__

    def __len__(self):
        return 1

    @abstractmethod
    def _compute_coeffs(self):
        pass

    @abstractmethod
    def sort_key(self):
        pass

    @staticmethod
    def __format_index(prefix, idx, show_index):
        if show_index:
            return f"{prefix}{idx}="
        else:
            return ''

    def get_sos(self) -> list[list[float]] | None:
        return [np.concatenate((self.b, self.a)).tolist()]


class Gain(Biquad):
    def __init__(self, fs, gain, f_id=-1):
        self.gain = gain
        super().__init__(fs, f_id=f_id)

    @property
    def filter_type(self):
        return 'Gain'

    @property
    def display_name(self):
        return 'Gain'

    def _compute_coeffs(self):
        return np.array([1.0, 0.0, 0.0]), np.array([10.0 ** (self.gain / 20.0), 0.0, 0.0])

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return Gain(new_fs, self.gain, f_id=self.id)

    def sort_key(self):
        return f"00000{self.gain:05}{self.filter_type}"

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'gain': self.gain
        }


class BiquadWithQ(Biquad):
    def __init__(self, fs, freq, q, f_id=-1):
        self.freq = round(float(freq), 2)
        self.q = float(q)
        self.w0 = 2.0 * math.pi * freq / fs
        self.cos_w0 = math.cos(self.w0)
        self.sin_w0 = math.sin(self.w0)
        self.alpha = self.sin_w0 / (2.0 * self.q)
        super().__init__(fs, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.freq == o.freq

    @property
    def description(self):
        return super().description + f" {self.freq}/{self.q:.4g}"

    def sort_key(self):
        return f"{self.freq:05}00000{self.filter_type}"


class Passthrough(Gain):
    def __init__(self, fs=1000, f_id=-1):
        super().__init__(fs, 0, f_id=f_id)

    @property
    def display_name(self):
        return 'Passthrough'

    @property
    def description(self):
        return 'Passthrough'

    def sort_key(self):
        return "ZZZZZZZZZZZZZZ"

    def resample(self, new_fs):
        return Passthrough(fs=new_fs, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs
        }


class BiquadWithQGain(BiquadWithQ):
    def __init__(self, fs, freq, q, gain, f_id=-1):
        self.gain = round(float(gain), 3)
        super().__init__(fs, freq, q, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.gain == o.gain

    @property
    def description(self):
        return super().description + f"/{self.gain}dB"

    def sort_key(self):
        return f"{self.freq:05}{self.gain:05}{self.filter_type}"


class PeakingEQ(BiquadWithQGain):
    '''
    H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)

            b0 =   1 + alpha*A
            b1 =  -2*cos(w0)
            b2 =   1 - alpha*A
            a0 =   1 + alpha/A
            a1 =  -2*cos(w0)
            a2 =   1 - alpha/A
    '''

    def __init__(self, fs, freq, q, gain, f_id=-1):
        super().__init__(fs, freq, q, gain, f_id=f_id)

    @property
    def filter_type(self):
        return 'PEQ'

    @property
    def display_name(self):
        return 'PEQ'

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([1.0 + self.alpha / A, -2.0 * self.cos_w0, 1.0 - self.alpha / A], dtype=np.float64)
        b = np.array([1.0 + self.alpha * A, -2.0 * self.cos_w0, 1.0 - self.alpha * A], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return PeakingEQ(new_fs, self.freq, self.q, self.gain, f_id=self.id)

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq,
            'q': self.q,
            'gain': self.gain
        }


def q_to_s(q, gain):
    '''
    translates Q to S for a shelf filter.
    :param q: the Q.
    :param gain: the gain.
    :return: the S.
    '''
    return 1.0 / ((((1.0 / q) ** 2.0 - 2.0) / (
            (10.0 ** (gain / 40.0)) + 1.0 / (10.0 ** (gain / 40.0)))) + 1.0)


def s_to_q(s, gain):
    '''
    translates S to Q for a shelf filter.
    :param s: the S.
    :param gain: the gain.
    :return: the Q.
    '''
    A = 10.0 ** (gain / 40.0)
    return 1.0 / math.sqrt(((A + 1.0 / A) * (1.0 / s - 1.0)) + 2.0)


class Shelf(BiquadWithQGain):
    def __init__(self, fs, freq, q, gain, count, f_id=-1):
        self.A = 10.0 ** (gain / 40.0)
        super().__init__(fs, freq, q, gain, f_id=f_id)
        self.count = count
        self.__cached_cascade = None

    def q_to_s(self):
        '''
        :return: the filter Q as S
        '''
        return q_to_s(self.q, self.gain)

    def __len__(self):
        return self.count

    def flatten(self):
        '''
        :return: an iterable of length count of this shelf where each shelf has count=1
        '''
        if self.count == 1:
            return [self]
        else:
            return [self.__class__(self.fs, self.freq, self.q, self.gain, 1)] * self.count

    def format_biquads(self, minidsp_style, separator=',\n', show_index=True, to_hex=False, fixed_point=False):
        single = super().format_biquads(minidsp_style, separator=separator, show_index=show_index,
                                        to_hex=to_hex, fixed_point=fixed_point)
        if self.count == 1:
            return single
        elif self.count > 1:
            return single * self.count
        else:
            raise ValueError('Shelf must have non zero count')

    def get_sos(self):
        return super().get_sos() * self.count

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'fs': self.fs,
            'fc': self.freq,
            'q': self.q,
            'gain': self.gain,
            'count': self.count
        }

    @property
    def description(self):
        if self.count > 1:
            return super().description + f" x{self.count}"
        else:
            return super().description


class LowShelf(Shelf):
    '''
    lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)

            b0 =    A*( (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha )
            b1 =  2*A*( (A-1) - (A+1)*cos(w0)                   )
            b2 =    A*( (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha )
            a0 =        (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha
            a1 =   -2*( (A-1) + (A+1)*cos(w0)                   )
            a2 =        (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha
    '''

    def __init__(self, fs, freq, q, gain, count=1, f_id=-1):
        super().__init__(fs, freq, q, gain, count, f_id=f_id)

    @property
    def filter_type(self):
        return 'LS'

    @property
    def display_name(self):
        return 'Low Shelf'

    def _compute_coeffs(self):
        A = 10.0 ** (self.gain / 40.0)
        a = np.array([
            (A + 1) + ((A - 1) * self.cos_w0) + (2.0 * math.sqrt(A) * self.alpha),
            -2.0 * ((A - 1) + ((A + 1) * self.cos_w0)),
            (A + 1) + ((A - 1) * self.cos_w0) - (2.0 * math.sqrt(A) * self.alpha)
        ], dtype=np.float64)
        b = np.array([
            A * ((A + 1) - ((A - 1) * self.cos_w0) + (2.0 * math.sqrt(A) * self.alpha)),
            2.0 * A * ((A - 1) - ((A + 1) * self.cos_w0)),
            A * ((A + 1) - ((A - 1) * self.cos_w0) - (2 * math.sqrt(A) * self.alpha))
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return LowShelf(new_fs, self.freq, self.q, self.gain, self.count, f_id=self.id)


class HighShelf(Shelf):
    '''
    highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)

                b0 =    A*( (A+1) + (A-1)*cos(w0) + 2*sqrt(A)*alpha )
                b1 = -2*A*( (A-1) + (A+1)*cos(w0)                   )
                b2 =    A*( (A+1) + (A-1)*cos(w0) - 2*sqrt(A)*alpha )
                a0 =        (A+1) - (A-1)*cos(w0) + 2*sqrt(A)*alpha
                a1 =    2*( (A-1) - (A+1)*cos(w0)                   )
                a2 =        (A+1) - (A-1)*cos(w0) - 2*sqrt(A)*alpha

    '''

    def __init__(self, fs, freq, q, gain, count=1, f_id=-1):
        super().__init__(fs, freq, q, gain, count, f_id=f_id)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o) and self.count == o.count

    @property
    def filter_type(self):
        return 'HS'

    @property
    def display_name(self):
        return 'High Shelf'

    def _compute_coeffs(self):
        A = self.A
        cos_w0 = self.cos_w0
        alpha = self.alpha
        a = np.array([
            (A + 1) - ((A - 1) * cos_w0) + (2.0 * math.sqrt(A) * alpha),
            2.0 * ((A - 1) - ((A + 1) * cos_w0)),
            (A + 1) - ((A - 1) * cos_w0) - (2.0 * math.sqrt(A) * alpha)
        ], dtype=np.float64)
        b = np.array([
            A * ((A + 1) + ((A - 1) * cos_w0) + (2.0 * math.sqrt(A) * alpha)),
            -2.0 * A * ((A - 1) + ((A + 1) * cos_w0)),
            A * ((A + 1) + ((A - 1) * cos_w0) - (2.0 * math.sqrt(A) * alpha))
        ], dtype=np.float64)
        return a / a[0], b / a[0]

    def resample(self, new_fs):
        '''
        Creates a filter at the specified fs.
        :param new_fs: the new fs.
        :return: the new filter.
        '''
        return HighShelf(new_fs, self.freq, self.q, self.gain, self.count, f_id=self.id)


class ComplexFilter(SOS, Sequence):
    '''
    A filter composed of many other filters.
    '''

    def __init__(self, fs=1000, filters=None, description='Complex', preset_idx=-1, listener=None, f_id=-1,
                 sort_by_id: bool = False):
        super().__init__(f_id=f_id, fs=fs)
        self.filters = [f for f in filters if f] if filters is not None else []
        self.__sort_by_id = sort_by_id
        self.description = description
        self.listener = listener
        self.preset_idx = preset_idx

    def __getitem__(self, i):
        return self.filters[i]

    def __len__(self):
        return len(self.filters)

    def __repr__(self):
        return self.description

    def __eq__(self, o: object) -> bool:
        equal = self.__class__.__name__ == o.__class__.__name__
        equal &= self.description == o.description
        equal &= self.id == o.id
        equal &= self.filters == o.filters
        return equal

    def child_names(self):
        return [x.__repr__() for x in self.filters]

    @property
    def sort_by_id(self) -> bool:
        return self.__sort_by_id

    @property
    def filter_type(self):
        return 'Complex'

    def format_biquads(self, invert_a, separator=',\n', show_index=True, to_hex=False, fixed_point=False):
        '''
        Formats the filter into a biquad report.
        :param fixed_point: if true, output biquads in fixed point format.
        :param to_hex: convert the biquad to a hex format (for minidsp).
        :param separator: separator biquads with the string.
        :param show_index: whether to include the biquad index.
        :param invert_a: whether to invert the a coeffs.
        :return: the report.
        '''
        import itertools
        return list(itertools.chain(*[f.format_biquads(invert_a,
                                                       separator=separator,
                                                       show_index=show_index,
                                                       to_hex=to_hex,
                                                       fixed_point=fixed_point)
                                      for f in self.filters]))

    def get_sos(self):
        ''' outputs the filter in cascaded second order sections ready for consumption by sosfiltfilt '''
        return [x for f in self.filters for x in f.get_sos()]

    def to_json(self):
        return {
            '_type': self.__class__.__name__,
            'description': self.description,
            'fs': self.fs,
            'filters': [x.to_json() for x in self.filters]
        }


class RejectionReason(IntEnum):
    RMS_EXCEEDED = 1
    MAX_EXCEEDED = 2
    BOTH_EXCEEDED = 3
    COSINE_TOO_LOW = 4
    DERIVATIVE_TOO_HIGH = 5


# ------------------------------
# Data classes
# ------------------------------
@dataclass
class BEQFilter:
    mag_freqs: np.ndarray
    mag_db: np.ndarray
    entry: CatalogueEntry


@dataclass
class BEQComposite:
    shape: np.ndarray
    assigned_indices: list[int] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    max_deltas: list[float] = field(default_factory=list)
    derivative_deltas: list[float] = field(default_factory=list)
    cosine_similarities: list[float] = field(default_factory=list)
    worst_rms_index: int | None = None
    worst_max_index: int | None = None
    fan_envelopes: list[np.ndarray] = field(default_factory=list)


@dataclass
class AssignmentRecord:
    entry_index: int
    assigned_composite: int | None = None
    rejected: bool = False
    rejection_reason: RejectionReason | None = None
    rms_value: float | None = None
    max_value: float | None = None
    derivative_value: float | None = None
    cosine_value: float | None = None


@dataclass
class BEQCompositePipelineResult:
    composites: list[BEQComposite]
    assignment_table: dict[int, AssignmentRecord]
    rms_limit: float
    max_limit: float
    cosine_limit: float
    derivative_limit: float


# ------------------------------
# Helper functions
# ------------------------------
def rms(a: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is not None:
        return float(np.sqrt(np.mean((a * weights) ** 2)))
    return float(np.sqrt(np.mean(a ** 2)))


def derivative_rms(a: np.ndarray) -> float:
    return rms(np.diff(a))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm != 0 else 1.0
