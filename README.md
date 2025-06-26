# Async Audio Playback Engine

## Overview

This project is an **asynchronous audio playback engine** designed for efficient and flexible sound processing in Python.  
It supports WAV audio format only.

---

## Dependencies

This library depends on the following third-party Python packages:

- PyAudio
- NumPy
- SciPy
- greenlet
- asyncio (standard library)

Make sure to install them before use.

---

## Usage Example

```python
from audiosystem import *

driver, loop, scheduler, sound = build_system(40000)
effecter = AudioEffecter(driver)
sound.add("base_audio.wav", "main")
audio_map1 = AudioMap(driver, len(sound["main"]))
effecter.inversion("main", "main2")
effecter.gain("main", 150, "main3")
audio_map1.write("main3", 0)
audio_map1.write("main2", 0)

@loop.task
async def task():
    await scheduler.play_soon(audio_map1.mix())
    scheduler.set_tps(20)
    async for _ in loop:
        print(loop.get_volume())

# run
loop.execute()
````

---

## Notes

* Currently supports **only WAV audio files**.
* This engine leverages Python's `asyncio` for asynchronous task handling.

---

## License

This project is licensed under the MIT License.

It includes the following third-party libraries with their respective licenses:

* PyAudio (MIT)
* NumPy (BSD 3-Clause)
* SciPy (BSD 3-Clause)
* greenlet (MIT)
* Python standard library (PSF License)

Please refer to their respective licenses for details.
