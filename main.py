from pyaudio import PyAudio, paInt16, paInt8
import numpy as np
from enum import Enum
from greenlet import greenlet
from scipy.io.wavfile import read
from scipy.signal import resample
from typing import Callable, Optional, Self, List, Dict, Union, overload, Literal, Tuple, SupportsInt, Union, Coroutine, Any, Set, TypeAlias
from time import perf_counter
from collections import defaultdict
from uuid import uuid4
import asyncio

_audio_group: TypeAlias = Union[np.ndarray, str, "_SoundData"]

class AudioFormat(Enum):
    """音声出力のフォーマットクラス"""
    int16 = (paInt16, np.int16)
    int8 = (paInt8, np.int8)

class DriverConfig:
    """エンジンの設定クラス"""
    def __init__(self, format: AudioFormat, rate: Union[int, SupportsInt] = 44100, chunk: int = 2000) -> None:
        self.format, self.rate = format, rate
        self.chunk = chunk

class Driver:
    """エンジンの基本クラス"""
    def __new__(cls, *a, **kw):
        if __name__ in _AssistManager.drivers:
            raise RuntimeError(f"An audio driver must be assigned at most once per thread, but this thread has been assigned {len(_AssistManager.drivers) + 1} drivers: {__name__}")
        return super().__new__(cls)
    def __init__(self, config: DriverConfig) -> None:
        self.audio = PyAudio() if not is_test_mode else None
        self.config = config
        self.extensions = {}
        self.manager = AudioStreamManager(self)
        _AssistManager.drivers[__name__] = self
        self.closed = False
    def close(self):
        self.manager.audio_scheduler.loop.stop()
        self.manager.close()
        del _AssistManager.drivers[__name__]
        self.extensions = {}
        self.closed = True
    def get(self):
        return self.manager.audio_scheduler.loop, self.manager.audio_scheduler

class _AssistManager:
    drivers: Dict[str, Driver] = {}

class AudioStreamManager:
    """音声ストリームを管理するクラス"""
    def __init__(self, driver: Driver):
        self.driver = driver
        self.stream = driver.audio.open(self.driver.config.rate, 2, self.driver.config.format.value[0], output=True) if not is_test_mode else None
        self.audio_scheduler = AudioScheduler(self)
    def close(self):
        if not is_test_mode and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.driver.audio.terminate()

class AudioScheduler:
    """音声をスケジュールするクラス"""
    def __init__(self, stream: AudioStreamManager) -> None:
        self.stream = stream
        self.chunks: List[Dict[str, List[np.ndarray]]] = [{} for _ in range(50)]
        self.chunkId = 0
        self.tps = -1
        self.loop = EventLoopScheduler(stream, self)
    def set_tps(self, tps: int):
        """for _ in loopのtpsを設定する。-1で無制限"""
        self.tps = tps
    def close(self):
        self.chunks = []
    async def _write(self, start_index: int, original_data: Union[np.ndarray, "_SoundData"], chunk_size: int = 100, is_fast: bool = True, id: Optional[str] = None):
        """
        chunks     : チャンクのリスト（各要素はchunk_sizeのNumPy配列）
        start_index: 配列全体を連続と見なしたときの開始インデックス
        data       : 書き込むNumPy配列
        chunk_size : 各チャンクのサイズ（デフォルト100）
        """
        if isinstance(original_data, _SoundData):
            data = original_data.get()
        else:
            data = original_data
        data: np.ndarray

        total_length = len(data)
        num_chunks_needed = (total_length + chunk_size - 1) // chunk_size

        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("ステレオ（2チャンネル）の (N, 2) 配列が必要です")

        num_samples = data.shape[0]
        pad_len = (chunk_size - (num_samples % chunk_size)) % chunk_size
        if pad_len > 0:
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')  # ← 2Dパディング

        # (N, 2) → (chunk数, chunk_size, 2)
        reshaped = data.reshape(-1, chunk_size, 2)

        # UUID とチャンクID の準備
        id = id or uuid4().hex
        chunks = self.chunks
        write_positions = np.arange(num_chunks_needed) + (start_index // chunk_size)
        write_chunk_ids = (write_positions + self.chunkId + 1) % len(chunks)

        # すべてのチャンクへまとめて登録（非同期性を壊さず）
        batch_buffers = defaultdict(list)  # IDごとにまとめる
        # 1チャンク単位でIDごとにまとめて登録
        for chunk_data, chunk_id in zip(reshaped, write_chunk_ids):
            batch_buffers[(chunk_id, id)].append(chunk_data)
        # まとめて chunks に登録
        current_chunk_id = None
        current_chunk: Optional[list] = None
        for i, ((chunk_id, id_key), buffers) in enumerate(batch_buffers.items()):
            if chunk_id != current_chunk_id:
                current_chunk = chunks[chunk_id]
                current_chunk_id = chunk_id
            if id_key not in current_chunk:
                current_chunk[id_key] = []
            current_chunk[id_key].extend(buffers)
            if not is_fast:
                if i + 1 % 4 == 0:
                    await asyncio.sleep(self.driver.config.chunk / self.driver.config.rate * 4)
        return AudioWriteHandler(self, range(0, 50), id, data)
    @overload
    async def play_later(self, seconds: float, data: _audio_group) -> "AudioWriteHandler": ...
    @overload
    async def play_later(self, seconds: float, data: "RealtimePipe") -> "PipeHandler": ...
    async def play_later(self, seconds: float, data: Union[_audio_group, "RealtimePipe"]):
        """秒数経過後に再生する"""
        if isinstance(data, RealtimePipe):
            handler = PipeHandler(self, data)
            self.loop.loop.call_later(seconds, handler._start)
            return handler
        data = data if isinstance(data, _SoundData) else data if isinstance(data, np.ndarray) else _SoundData(self.stream.driver.extensions["SoundsManager"], data)
        return await self._write(round(seconds * self.stream.driver.config.rate), data, self.stream.driver.config.chunk)
    @overload
    async def play_soon(self, data: _audio_group) -> "AudioWriteHandler": ...
    @overload
    async def play_soon(self, data: "RealtimePipe") -> "PipeHandler": ...
    async def play_soon(self, data: Union[_audio_group, "RealtimePipe"]):
        """1ティック後に再生"""
        if isinstance(data, RealtimePipe):
            handler = PipeHandler(self, data)
            handler._start()
            return handler
        data = data if isinstance(data, _SoundData) else data if isinstance(data, np.ndarray) else _SoundData(self.stream.driver.extensions["SoundsManager"], data)
        return await self._write(0, data, self.stream.driver.config.chunk)

class AudioWriteHandler:
    """書き込んだ音声をキャンセルできる制御クラス"""
    def __init__(self, scheduler: AudioScheduler, written: range, id: str, data: np.ndarray):
        self.id, self.written, self.scheduler = id, written, scheduler
        self.canceled = False
        self.callback_function = None
        self.play_waiter = self.scheduler.loop.loop.create_future()
        self.callback_caller = self.scheduler.loop.loop.call_later(len(data) / self.scheduler.stream.driver.config.rate, self._callback_runner)
    def _callback_runner(self):
        if self.callback_function is not None:
            self.scheduler.loop.loop.create_task(self.callback_function())
        self.is_played = True
        self.play_waiter.set_result(True)
    def callback(self, func: Callable[[], Coroutine]):
        self.callback_function = func
        return func
    def wait(self):
        return self.play_waiter
    def cancel(self):
        self.play_waiter.set_result(False)
        """書き込んだ音声をキャンセルする。"""
        for i in self.written:
            self.scheduler.chunks[i].pop(self.id, None)
        self.canceled = True

class PipeHandler:
    def __init__(self, scheduler: AudioScheduler, pipe: "RealtimePipe"):
        self.pipe = pipe
        self.scheduler = scheduler
        self._writer: Optional[asyncio.Task] = None
        self._last_written_chunk: Optional[asyncio.Future] = None
        self._id = uuid4().hex
        self._tasks = set()
        self.canceled = False
        self._ids = set()
    def _start(self):
        self.pipe._handler = self
        async def writer():
            try:
                while True:
                    data = await self.pipe._pipe.get()
                    max_len = len(data)
                    datas = [data]
                    if not self.pipe._pipe.empty():
                        while not self.pipe._pipe.empty():
                            data = await self.pipe._pipe.get()
                            if max_len < len(data):
                                max_len = len(data)
                            datas.append(data)
                    map = AudioMap(max_len)
                    for data in datas:
                        map.write(data, 0)
                    id = uuid4().hex
                    self._ids.add(id)
                    task = asyncio.create_task(self.scheduler._write(0, map.mix(), self.scheduler.stream.driver.config.chunk, False, id))
                    map.close()
                    self.scheduler.loop.tasks.add(task)
                    self._tasks.add(task)
                    self._last_written_chunk = task
            except asyncio.CancelledError:
                pass
        self._writer = task = asyncio.create_task(writer())
        self.scheduler.loop.active_pipes[self._id] = task
    def cancel(self):
        self.canceled = True
        async def _func():
            tasks = asyncio.gather(*self._tasks)
            tasks.cancel()
            await tasks
            for i in range(0, 50):
                for id in self._ids:
                    self.scheduler.chunks[i].pop(id, None)
        return asyncio.create_task(_func())

class AsyncConnectedEventLoop(asyncio.SelectorEventLoop):
    def __init__(self, driver: Driver) -> None:
        super().__init__()
        self.driver = driver
        self.last_played_timer = 0.0
    def _run_once(self):
        self.last_played_timer = perf_counter()
        super()._run_once()
    def run_until_complete(self, future):
        async def func():
            task = self.create_task(future)
            iter(self.driver.manager.audio_scheduler.loop)
            try:
                while True:
                    try:
                        if len(self.driver.manager.audio_scheduler.chunks[self.driver.manager.audio_scheduler.chunkId % 50]) == 0: # もしも今のチャンクに音声がないなら、スキップ + そのチャンク分スリープ
                            if self.driver.manager.audio_scheduler.loop.is_stopping:
                                break # 今のチャンクに音声がなくても、終了フラグがたっているなら終了
                            self.driver.manager.audio_scheduler.chunkId += 1
                            self.driver.manager.audio_scheduler.loop._tick_up()
                            await asyncio.sleep(self.driver.config.chunk / self.driver.config.rate + ((self.driver.manager.audio_scheduler.loop.last_time + 1 / self.driver.manager.audio_scheduler.tps) - perf_counter()))
                        else:
                            next(self.driver.manager.audio_scheduler.loop)
                            await asyncio.sleep(0)
                    except StopIteration:
                        break
            finally:
                task.cancel()
                await task
                for task in self.driver.manager.audio_scheduler.loop.tasks:
                    if not task.done():
                        task.cancel()
                        await task
                for task in list(self.driver.manager.audio_scheduler.loop.active_pipes.values()):
                    if not task.done():
                        task.cancel()
                        await task
                self.stop()
        return super().run_until_complete(func())

class EventLoopScheduler:
    """音を再生するループ"""
    def __init__(self, stream: AudioStreamManager, scheduler: AudioScheduler):
        self.tick: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None
        self.player_thread: Optional[greenlet] = None
        self.tick_thread: Optional[greenlet] = None
        self.stream = stream
        self.scheduler = scheduler
        self.tick_sync_condition = asyncio.Event()
        self.last_time = perf_counter()
        self.is_stopping = False
        self.tick_count = 0
        self.now_playing_audio = np.zeros((self.stream.driver.config.chunk, 2))
        self.loop = AsyncConnectedEventLoop(self.scheduler.stream.driver)
        self.tick_callbacks: Dict[int, Set[Callable[[], Coroutine[Any, Any, Any]]]] = {}
        self.tasks = set()
        self.active_pipes: Dict[str, asyncio.Task] = {}
        self.chunk_play_queue = asyncio.Queue()
        self.stopped = False
    def _audioPlayer(self):
        while True:
            if is_test_mode:
                break
            array = self.scheduler.chunks[self.scheduler.chunkId % len(self.scheduler.chunks)]
            if len(array) != 0:
                sum_data = np.sum([data.pop(0) for data in array.values() if len(data) > 0], axis=0, dtype=np.int64) / len(array)
                data = sum_data.astype(self.stream.driver.config.format.value[1])
                self.now_playing_audio = data
                self.stream.stream.write(data.tobytes())
                self.scheduler.chunkId = (self.scheduler.chunkId + 1) % len(self.scheduler.chunks)
            self.tick_thread.switch()
    def task(self, func: Callable[[], Coroutine[Any, Any, Any]]) -> Callable[[], Coroutine[Any, Any, Any]]:
        """loop.execute()時に実行する関数を指定するデコレーター"""
        async def tick():
            try:
                await func()
            except asyncio.CancelledError:
                pass
            finally:
                self.scheduler.stream.close()
        self.tick = tick
        return func
    def execute(self):
        """タスクを起動"""
        def main():
            try:
                self.loop.run_until_complete(self.tick())
            finally:
                self.loop.close()
        self.tick_thread = greenlet(main)
        try:
            self.tick_thread.switch()
        except KeyboardInterrupt:
            pass
    def __iter__(self) -> Self:
        """イベントループの起動。"""
        self.player_thread = greenlet(self._audioPlayer, self.tick_thread)
        return self
    def __aiter__(self) -> Self:
        return self
    async def __anext__(self) -> None:
        await self.tick_sync_condition.wait()
        self.tick_sync_condition.clear()
        if self.is_stopping:
            raise StopAsyncIteration
        return self.tick_count
    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.is_stopping = True
    def __next__(self) -> None:
        """1ティック進める"""
        if self.is_stopping:
            raise StopIteration
        i = 0
        if self.scheduler.tps != -1:
            if self.last_time + 1 / self.scheduler.tps > perf_counter():
                while self.last_time + 1 / self.scheduler.tps > perf_counter():
                    i += 1
                    self.player_thread.switch()
        self.tick_sync_condition.set()
        self.player_thread.switch()
        for tick in list(self.tick_callbacks.keys()):
            if tick <= self.tick_count:
                for callback in self.tick_callbacks[tick]:
                    self.tasks.add(self.loop.create_task(callback()))
                del self.tick_callbacks[tick]
            else:
                break
        self.tick_count += 1
        if self.scheduler.tps != -1:
            self.last_time = perf_counter()
    def _tick_up(self):
        self.tick_sync_condition.set()
        for tick in list(self.tick_callbacks.keys()):
            if tick <= self.tick_count:
                for callback in self.tick_callbacks[tick]:
                    self.tasks.add(self.loop.create_task(callback()))
                del self.tick_callbacks[tick]
            else:
                break
        self.tick_count += 1
        if self.scheduler.tps != -1:
            self.last_time = perf_counter()
    def get_volume(self):
        volume = np.mean(np.abs(self.now_playing_audio.mean(axis=1)))
        return volume
    def execute_to_tick(self, ticks: int):
        def Wrapper(func: Callable[[], Coroutine[Any, Any, Any]]):
            if ticks + self.tick_count in self.tick_callbacks:
                self.tick_callbacks[ticks + self.tick_count].add(func)
            else:
                self.tick_callbacks[ticks + self.tick_count] = {func}
            return func
        return Wrapper
    def execute_to_times(self, seconds: Optional[int] = None, minutes: Optional[int] = None, hours: Optional[int] = None) -> Callable[[Callable[[], Coroutine[Any, Any, Any]]], asyncio.Handle]:
        if seconds is minutes is hours is None:
            raise ValueError("To use this function, you must specify one of the time arguments.")
        real_time = 0
        if seconds:
            real_time += seconds
        if minutes:
            real_time += minutes * 60
        if hours:
            real_time += hours * 3600
        def Wrapper(func: Callable[[], Coroutine[Any, Any, Any]]) -> asyncio.Handle:
            return self.loop.call_later(real_time, lambda: self.loop.create_task(func()))
        return Wrapper

class _ExtensionBaseMeta(type):
    def __call__(self, *args, **kwds):
        if __name__ not in _AssistManager.drivers:
            raise RuntimeError("The extension was initialized even though the driver was not created.")
        obj =  super().__call__(*args, **kwds)
        obj.driver = driver = _AssistManager.drivers[__name__]
        driver.extensions[type(obj).__name__] = obj
        return obj

class ExtensionBase(metaclass=_ExtensionBaseMeta):
    driver: Driver
    """拡張機能のベースクラス"""

class SoundsManager(ExtensionBase):
    """音マテリアルを管理するクラス。"""
    def __init__(self):
        super().__init__()
        self.sounds = {}
    @overload
    def add(self, filename: np.ndarray, sound_as: str, rate: int): ...
    @overload
    def add(self, filename: str, sound_as: str): ...
    def add(self, filename: str, sound_as: str, rate: Optional[int] = None):
        """音の追加
        例: `sound_manager.add("audio.wav", "audio")`"""
        if isinstance(filename, np.ndarray):
            data = filename
        else:
            rate, data = read(filename)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        if self.driver.config.rate != rate:
            data = resample(data, round(len(data) * self.driver.config.rate / rate)) # pyright: ignore
        data: np.ndarray
        if data.dtype != self.driver.config.format.value[1]:
            target_max = np.iinfo(self.driver.config.format.value[1]).max
            now_max = np.max(data)
            scaling = target_max / now_max
            data *= scaling
            data = data.astype(self.driver.config.format.value[1])
        self.sounds[sound_as] = data
    def __getitem__(self, name: str):
        """保存された音を取得"""
        return _SoundData(self, name)

class _SoundData:
    def __init__(self, manager: SoundsManager, id: str) -> None:
        self.manager, self.id = manager, id
    def get(self):
        return self.manager.sounds[self.id]
    def __len__(self):
        return len(self.get())

class AudioEffecter(ExtensionBase):
    """音声にエフェクトを掛ける基本的な機能
    拡張依存: SoundsManager"""
    def __init__(self):
        super().__init__()
        self.manager: SoundsManager = _AssistManager.drivers[__name__].extensions["SoundsManager"]
    def sum(self, *audios: Union[_SoundData, str], save_as: str):
        """音を合成する"""
        audios = tuple(audio if isinstance(audio, _SoundData) else _SoundData(self.driver.extensions["SoundsManager"], audio) for audio in audios)
        self.manager.sounds[save_as] = np.sum([audio.get() for audio in audios], axis=0, dtype=np.int64) / len(audios)
    def reverse(self, audio: Union[_SoundData, str], save_as: str):
        """音を反転させる"""
        audio = audio if isinstance(audio, _SoundData) else _SoundData(self.driver.extensions["SoundsManager"], audio)
        self.manager.sounds[save_as] = audio.get()[::-1]
    def inversion(self, audio: Union[_SoundData, str], save_as: str):
        """音の波を反転させる"""
        audio = audio if isinstance(audio, _SoundData) else _SoundData(self.driver.extensions["SoundsManager"], audio)
        self.manager.sounds[save_as] = audio.get() * -1
    def gain(self, audio: Union[_SoundData, str], gain: float, save_as: str):
        """音の音量を変える
        gain: 100分率"""
        audio = audio if isinstance(audio, _SoundData) else _SoundData(self.driver.extensions["SoundsManager"], audio)
        self.manager.sounds[save_as] = audio.get() * (gain / 100)

class _AudioWriteHandlerForAudioMap:
    def __init__(self, map: "AudioMap", id: str):
        self.map = map
        self.id = id
    def cancel(self):
        del self.map.data[self.id]

class AudioMap:
    """システムマップに直接書きこまない仮想オーディオマップ
    size: サンプル数"""
    def __init__(self, size: int):
        super().__init__()
        self.rate = _AssistManager.drivers[__name__].config.rate
        self.size = slice(0, size)
        self.data: Dict[str, np.ndarray] = {}
    def write(self, audio: _audio_group, start: float, end: Optional[float] = None):
        """仮想オーディオマップに書き込む
        start/end: サンプル数"""
        end = end if end is not None else self.size.stop
        start = start
        if self.size.start < start or (self.size.stop < end and end != -1):
            raise ValueError("The capacity set for start or end has been exceeded.")
        buffer = np.zeros((self.size.stop, 2))
        data = audio if isinstance(audio, _SoundData) else audio if isinstance(audio, np.ndarray) else _SoundData(self.driver.extensions["SoundsManager"], audio)
        if isinstance(data, _SoundData):
            data = data.get()
        buffer[start:end, ...] = data
        id = uuid4().hex
        self.data[id] = buffer
        return _AudioWriteHandlerForAudioMap(self, id)
    def mix(self):
        return np.sum(list(self.data.values()), axis=0)
    def close(self):
        self.data = {}

class AudioPipeline(ExtensionBase):
    def __init__(self, *pipelines: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.pipelines = pipelines
    def execute(self, audio: _audio_group):
        data = audio.get() if isinstance(audio, _SoundData) else audio if isinstance(audio, np.ndarray) else self.driver.extensions["SoundsManager"].sounds[audio] if isinstance(audio, str) else None
        for pipe in self.pipelines:
            data = pipe(data)
        return data

class _SystemBuilder:
    def __init__(self, rate: Union[int, SupportsInt], auto_execute, format: AudioFormat = AudioFormat.int16):
        config = DriverConfig(format, rate, 400)
        self._driver = Driver(config)
        self._loop, self._scheduler = self._driver.get()
        self._sounds_manager = SoundsManager()
        self._auto = auto_execute
    def __enter__(self):
        return (self._driver, self._loop, self._scheduler, self._sounds_manager)
    def __exit__(self, exc_type, exc_value, tb):
        if self._auto:
            if exc_type is not None:
                raise exc_value
            try:
                self._loop.execute()
            finally:
                self._driver.close()
        else:
            if not self._driver.closed:
                self._driver.close()

@overload
def build_system(rate: Union[SupportsInt, int], format: AudioFormat = AudioFormat.int16, *, is_context: Literal[False] = False) -> Tuple[Driver, EventLoopScheduler, AudioScheduler, SoundsManager]: ...

@overload
def build_system(rate: Union[SupportsInt, int], format: AudioFormat = AudioFormat.int16, *, is_context: Literal[True], auto_execute: bool = False) -> _SystemBuilder: ...

def build_system(rate: Union[SupportsInt, int], format: AudioFormat = AudioFormat.int16, *, is_context: bool = False, auto_execute: bool = False):
    """システムを初期化します。
    ## Args:
    - is_context: 戻り値をコンテキストマネージャーとして返します。
    - auto_execute: is_context引数がTrueの時に、loop.execute()を実行しなくても自動で実行することができます。"""
    if is_context:
        return _SystemBuilder(rate, auto_execute=auto_execute, format=format)
    else:
        config = DriverConfig(format, rate, 400)
        driver = Driver(config)
        loop, scheduler = driver.get()
        sounds_manager = SoundsManager()
        return (driver, loop, scheduler, sounds_manager)

class AudioPipeline(ExtensionBase):
    def __init__(self, *pipelines: Callable[[np.ndarray], np.ndarray]):
        super().__init__()
        self.pipelines = pipelines
    def execute(self, audio: _audio_group):
        data = audio.get() if isinstance(audio, _SoundData) else audio if isinstance(audio, np.ndarray) else self.driver.extensions["SoundsManager"].sounds[audio] if isinstance(audio, str) else None
        if data is None:
            raise TypeError(f"Expected type _SoundData, ndarray, or str, but got {type(audio).__name__}")
        for pipe in self.pipelines:
            data = pipe(data)
        return data

class RealtimePipe:
    def __init__(self):
        self._pipe = asyncio.Queue()
        self.driver = _AssistManager.drivers[__name__]
        self.closed = False
        self._handler: Optional[PipeHandler] = None
    async def write(self, data: _audio_group):
        data = data.get() if isinstance(data, _SoundData) else data if isinstance(data, np.ndarray) else _SoundData(self.driver.extensions["SoundsManager"], data)
        await self._pipe.put(data)
    def close(self):
        if self._handler is not None:
            if not self._handler.canceled:
                self._handler.cancel()
        self._pipe.task_done()
        self.closed = True

is_test_mode = False

def set_test_mode():
    global is_test_mode
    is_test_mode = True
