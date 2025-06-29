from pyaudio import PyAudio, paInt16, paInt8
import numpy as np
from enum import Enum
from greenlet import greenlet
from scipy.io.wavfile import read
from scipy.signal import resample
from typing import Callable, Optional, Self, List, Dict, Union, Coroutine, NoReturn, Tuple, Type, Any, ParamSpec
from time import time
from functools import lru_cache
from collections import defaultdict
from uuid import uuid4
import asyncio

class AudioFormat(Enum):
    """音声出力のフォーマットクラス"""
    int16 = (paInt16, np.int16)
    int8 = (paInt8, np.int8)

class DriverConfig:
    """エンジンの設定クラス"""
    def __init__(self, format: AudioFormat, rate: int = 44100, chunk: int = 2000) -> None:
        self.format, self.rate = format, rate
        self.chunk = chunk

class Driver:
    """エンジンの基本クラス"""
    def __init__(self, config: DriverConfig) -> None:
        self.audio = PyAudio()
        self.config = config
        self.extensions: Dict[str, ExtensionBase] = {}
        self.manager = AudioStreamManager(self)
    def get(self):
        return self.manager.audio_scheduler.loop, self.manager.audio_scheduler

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
        self.tps: int = -1
        self.loop = EventLoopScheduler(stream, self)
    def set_tps(self, tps: int):
        """for _ in loopのtpsを設定する。-1で無制限"""
        self.tps = tps
    async def _write(self, start_index: int, original_data: Union[np.ndarray, "_SoundData", str], chunk_size: int = 100):
        """
        chunks     : チャンクのリスト（各要素はchunk_sizeのNumPy配列）
        start_index: 配列全体を連続と見なしたときの開始インデックス
        data       : 書き込むNumPy配列
        chunk_size : 各チャンクのサイズ（デフォルト100）
        """
        if isinstance(original_data, _SoundData):
            data = original_data.get()
        elif isinstance(original_data, str):
            data = self.stream.driver.extensions["SoundsManager"].sounds[original_data]
        else:
            data = original_data

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
        id = uuid4().hex
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
        for (chunk_id, id_key), buffers in batch_buffers.items():
            if chunk_id != current_chunk_id:
                current_chunk = chunks[chunk_id]
                current_chunk_id = chunk_id
            if id_key not in current_chunk:
                current_chunk[id_key] = []
            current_chunk[id_key].extend(buffers)
        return AudioWriteHandler(self, range(write_chunk_ids[0], write_chunk_ids[-1] + 1), id, data)
    async def play_later(self, seconds: float, data: Union[np.ndarray, "_SoundData", str]):
        """秒数経過後に再生する"""
        data = data if isinstance(data, _SoundData) else data if isinstance(data, np.ndarray) else _SoundData(self.stream.driver.extensions["SoundsManager"], data)
        return await self._write(round(seconds * self.stream.driver.config.rate), data, self.stream.driver.config.chunk)
    async def play_soon(self, data: Union[np.ndarray, "_SoundData", str]):
        """1ティック後に再生"""
        data = data if isinstance(data, _SoundData) else data if isinstance(data, np.ndarray) else _SoundData(self.stream.driver.extensions["SoundsManager"], data)
        return await self._write(0, data, self.stream.driver.config.chunk)

class AudioWriteHandler:
    """書き込んだ音声をキャンセルできる制御クラス"""
    def __init__(self, scheduler: AudioScheduler, written: range, id: str, data: np.ndarray):
        self.id, self.written, self.scheduler = id, written, scheduler
        self.callback_function: Optional[Callable[[], Coroutine]] = None
        self.data = data
        self.is_played = False
        self.play_waiter = self.scheduler.loop.loop.create_future()
        self.callback_caller = self.scheduler.loop.loop.call_later(len(self.data) / self.scheduler.stream.driver.config.rate, self._callback_runner)
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
        """書き込んだ音声をキャンセルする。"""
        self.play_waiter.set_result(False)
        if not self.is_played:
            self.callback_caller.cancel()
        for i in self.written:
            self.scheduler.chunks[i].pop(self.id, None)

class PlayerStop(Exception):
    pass

class AsyncConnectedEventLoop(asyncio.SelectorEventLoop):
    def __init__(self, driver: Driver) -> None:
        super().__init__()
        self.driver = driver
        self.end_flag = False
        self.task: Optional[asyncio.Task] = None
    def _run_once(self):
        try:
            next(self.driver.manager.audio_scheduler.loop)
        except StopIteration:
            self.end_flag = True
            return
        super()._run_once()
    def stop(self):
        super().stop()
    def run_until_complete(self, future):
        def call_task():
            async def task():
                while True:
                    await asyncio.sleep(self.driver.config.chunk / self.driver.config.rate)
            self.task = self.create_task(task())
        self.call_soon(call_task)
        iter(self.driver.manager.audio_scheduler.loop)
        return super().run_until_complete(future)
    def __del__(self): pass

class EventLoopScheduler:
    """音を再生するループ"""
    def __init__(self, stream: AudioStreamManager, scheduler: AudioScheduler):
        self.tick: Optional[Callable[[], Coroutine]] = None
        self.player_thread: Optional[greenlet] = None
        self.tick_thread: Optional[greenlet] = None
        self.stream = stream
        self.scheduler = scheduler
        self.tick_sync_condition = asyncio.Event()
        self.last_time = time()
        self.is_stopping = False
        self.tick_count = 0
        self.now_playing_audio = np.zeros((self.stream.driver.config.chunk, 2))
        self.loop = AsyncConnectedEventLoop(self.scheduler.stream.driver)
        self.tick_callbacks: Dict[int, List[Callable[[], Coroutine]]] = {}
        self.is_called_stop = False
    def _audioPlayer(self):
        try:
            while True:
                if is_test_mode:
                    break
                array = self.scheduler.chunks[self.scheduler.chunkId % len(self.scheduler.chunks)]
                if len(array) != 0:
                    sum_data = np.sum([data.pop(0) for data in array.values() if len(data) > 0], axis=0, dtype=np.int64) / len(array)
                    data = sum_data.astype(self.stream.driver.config.format.value[1])
                    self.now_playing_audio = data
                    self.stream.stream.write(data.tobytes())
                    for key, data in array.items():
                        if not data:
                            del array[key]
                    self.scheduler.chunkId = (self.scheduler.chunkId + 1) % len(self.scheduler.chunks)
                self.tick_thread.switch()
        except PlayerStop:
            return
    def task(self, func: Callable[[], Coroutine]) -> Callable[[], Coroutine]:
        """loop.execute()時に実行する関数を指定するデコレーター"""
        async def tick():
            try:
                await func()
            finally:
                self.loop.task.cancel()
                await self.loop.task
                self.scheduler.stream.close()
                self.loop.stop()
        self.tick = tick
        return func
    def execute(self) -> Union[None, NoReturn]:
        """タスクを起動"""
        def main():
            try:
                self.loop.run_until_complete(self.tick())
            finally:
                if not self.is_called_stop:
                    self.stop()
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
    def stop(self) -> None:
        self.is_stopping = True
        self.is_called_stop = True
        self.player_thread.throw(PlayerStop)
    def __next__(self) -> None:
        """1ティック進める"""
        if self.is_stopping:
            raise StopIteration
        if self.scheduler.tps != -1:
            if self.last_time + 1 / self.scheduler.tps > time():
                while self.last_time + 1 / self.scheduler.tps > time():
                    self.player_thread.switch()
        self.tick_sync_condition.set()
        self.player_thread.switch()
        if self.tick_count in self.tick_callbacks:
            for callback in self.tick_callbacks[self.tick_count]:
                self.loop.create_task(callback())
            del self.tick_callbacks[self.tick_count]
        self.tick_count += 1
        if self.scheduler.tps != -1:
            self.last_time = time()
    def get_volume(self) -> float:
        volume = np.mean(np.abs(self.now_playing_audio.mean(axis=1)))
        return volume
    def execute_to_tick(self, ticks: int) -> Callable[[Callable[[], Coroutine[Any, Any, Any]]], Callable[[], Coroutine[Any, Any, Any]]]:
        def Wrapper(func: Callable[[], Coroutine]):
            if ticks in self.tick_callbacks:
                self.tick_callbacks[ticks + self.tick_count].append(func)
            else:
                self.tick_callbacks[ticks + self.tick_count] = [func]
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


class ExtensionBase:
    """拡張機能のベースクラス"""
    def __init__(self, driver: Driver):
        driver.extensions[self.__class__.__name__] = self

class SoundsManager(ExtensionBase):
    """音マテリアルを管理するクラス。"""
    def __init__(self, driver: Driver):
        super().__init__(driver)
        self.driver = driver
        self.sounds = {}
    def add(self, filename: Union[str, np.ndarray], sound_as: str, rate: Optional[int] = None) -> None:
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
    def __getitem__(self, name: str) -> "_SoundData":
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
    def __init__(self, driver: Driver):
        super().__init__(driver)
        self.manager: SoundsManager = driver.extensions["SoundsManager"]
        self.driver = driver
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

class AudioMap(ExtensionBase):
    """システムマップに直接書きこまない仮想オーディオマップ
    size: サンプル数"""
    def __init__(self, driver: Driver, size: int):
        super().__init__(driver)
        self.driver = driver
        self.rate = self.driver.config.rate
        self.size = slice(0, size)
        self.data: Dict[str, np.ndarray] = {}
    def write(self, audio: Union[np.ndarray, _SoundData, str], start: float, end: Optional[float] = None) -> _AudioWriteHandlerForAudioMap:
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
    def mix(self) -> np.ndarray:
        return np.sum(list(self.data.values()), axis=0)

def build_system(rate: int, format: AudioFormat = AudioFormat.int16) -> Tuple[Driver, EventLoopScheduler, AudioScheduler, SoundsManager]:
    config = DriverConfig(format, rate, 400)
    driver = Driver(config)
    loop, scheduler = driver.get()
    sounds_manager = SoundsManager(driver)
    return driver, loop, scheduler, sounds_manager

class AudioPipeline(ExtensionBase):
    def __init__(self, driver: Driver, *pipelines: Callable[[np.ndarray], np.ndarray]):
        super().__init__(driver)
        self.pipelines = pipelines
        self.driver = driver
    def execute(self, audio: Union[np.ndarray, str, _SoundData]):
        data = audio.get() if isinstance(audio, _SoundData) else audio if isinstance(audio, np.ndarray) else self.driver.extensions["SoundsManager"].sounds[audio] if isinstance(audio, str) else None
        for pipe in self.pipelines:
            data = pipe(data)
        return data

is_test_mode = False

def set_test_mode():
    global is_test_mode
    is_test_mode = True
