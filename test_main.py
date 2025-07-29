from main import build_system, AudioMap, set_test_mode, AudioPipeline, AudioEffecter, ExtensionBase
from numpy import sin, cos, arange

set_test_mode()

driver, loop, scheduler, sound = build_system(40000)

def test_audio_map():
    audio_map = AudioMap(driver.config.rate * 4)

    audio1 = arange(0, driver.config.rate * 4)[:, None].repeat(2, axis=1)

    a, b = sin(audio1), cos(audio1)

    audio_map.write(a, 0)
    audio_map.write(b, 0)

    c = audio_map.mix()

    assert (c == a + b).all()


def test_pipeline():
    pipeline = AudioPipeline(
        lambda data: data + b,
        lambda data: data[::-1]
    )

    audio1 = arange(0, driver.config.rate * 4)[:, None].repeat(2, axis=1)

    a, b = sin(audio1), cos(audio1)

    data = pipeline.execute(a)

    assert (data == (a + b)[::-1]).all()


def test_effecter():
    effecter = AudioEffecter()

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
    ExtensionBase()

    assert "ExtensionBase" in driver.extensions

