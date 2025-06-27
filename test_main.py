from main import build_system, AudioMap, set_test_mode, AudioPipeline, AudioEffecter, ExtensionBase, PlayerStop
from numpy import sin, cos, arange
from pytest import raises
from time import perf_counter

set_test_mode()

def test_audio_map():
    driver, loop, scheduler, sound = build_system(40000)

    audio_map = AudioMap(driver, driver.config.rate * 4)

    audio1 = arange(0, driver.config.rate * 4)[:, None].repeat(2, axis=1)

    a, b = sin(audio1), cos(audio1)

    audio_map.write(a, 0)
    audio_map.write(b, 0)

    c = audio_map.mix()

    assert (c == a + b).all()

def test_pipeline():
    driver, loop, scheduler, sound = build_system(40000)

    pipeline = AudioPipeline(driver, 
        lambda data: data + b,
        lambda data: data[::-1]
    )

    audio1 = arange(0, driver.config.rate * 4)[:, None].repeat(2, axis=1)

    a, b = sin(audio1), cos(audio1)

    data = pipeline.execute(a)

    assert (data == (a + b)[::-1]).all()

def test_effecter():
    driver, loop, scheduler, sound = build_system(44100)
    effecter = AudioEffecter(driver)

    audio1 = arange(0, driver.config.rate * 4)[:, None].repeat(2, axis=1)

    a = sin(audio1)

    sound.add(a, "main", 44100)

    effecter.inversion("main", "main2")
    effecter.reverse("main", "main3")
    effecter.gain("main", 150, "main4")

    assert (sound["main2"].get() == sound["main"].get() * -1).all()
    assert (sound["main3"].get() == sound["main"].get()[::-1]).all()
    assert (sound["main4"].get() == sound["main"].get() * 1.5).all()

def test_extension_base():
    driver, loop, scheduler, sound = build_system(40000)

    ExtensionBase(driver)

    assert "ExtensionBase" in driver.extensions

def test_errors_write_monaural():
    driver, loop, scheduler, sound = build_system(40000)

    audio1 = arange(0, driver.config.rate * 4)

    a = sin(audio1)

    @loop.task
    async def task():
        with raises(ValueError):
            await scheduler.play_soon(a)
        with raises(PlayerStop):
            loop.stop()

def test_tps():
    driver, loop, scheduler, sound = build_system(40000)

    intervals = []
    timestamps = []

    @loop.task
    async def task():
        scheduler.set_tps(20)
        async for _ in loop:
            if len(timestamps) >= 400:
                return
            timestamps.append(perf_counter())
    with raises(PlayerStop):
        loop.execute()

    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i - 1])

    expected = 0.05
    tolerance = 0.05
    
    for i, interval in enumerate(intervals):
        assert abs(interval - expected) < tolerance, f"{i}番目の間隔が不正: {interval:.4f}秒"
