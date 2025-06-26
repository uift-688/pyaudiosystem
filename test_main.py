from main import build_system, AudioMap, set_test_mode
from numpy import linspace, sin, cos, sum

def test_all():
    set_test_mode()

    driver, loop, scheduler, sound = build_system(40000)

    audio_map = AudioMap(driver, driver.config.rate)

    audio1 = linspace(0, driver.config.rate, 1)

    a, b = sin(audio1), cos(audio1)

    audio_map.write(a, 0)
    audio_map.write(b, 0)

    c = audio_map.mix()

    assert (c == a + b).all()
