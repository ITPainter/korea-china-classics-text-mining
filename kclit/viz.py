"""그래프에 한글·한자를 표시하기 위한 글꼴 설정."""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

NOTO_CJK = "Noto Sans CJK JP"


def setup_korean_font() -> str:
    """Colab에서는 Noto CJK 글꼴을 설치하고 matplotlib 기본 글꼴로 정한다.

    Noto Sans CJK는 한글과 한자(번체·간체)를 모두 담고 있다.
    """
    names = {f.name for f in font_manager.fontManager.ttflist}
    if NOTO_CJK not in names and Path("/usr/bin/apt-get").exists():
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"], check=False, capture_output=True)
        for f in Path("/usr/share/fonts/opentype/noto").glob("NotoSansCJK-*.ttc"):
            font_manager.fontManager.addfont(str(f))
        names = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (NOTO_CJK, "Noto Sans KR", "AppleGothic", "Malgun Gothic"):
        if candidate in names:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    return plt.rcParams["font.family"][0]
