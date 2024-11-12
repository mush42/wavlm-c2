import os
import sys
from pathlib import Path

from ctranslate2.converters.transformers import main

from wavlm2c2 import WavLMLoader


if __name__ == '__main__':
    sys.argv.extend(["--model", "./microsoft-wavlm-large"])
    sys.argv.extend(["--output", "./wavlm-c2"])
    main()
