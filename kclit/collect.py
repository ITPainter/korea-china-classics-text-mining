"""자료 수집: 예의 바른 크롤링, 위키문헌 원문 받기, 고문헌 OCR (2장)."""

import re
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

USER_AGENT = "korea-china-classics-text-mining/0.1 (educational research code)"
_last_request: dict[str, float] = {}


def allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    parts = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
    try:
        rp.read()
    except OSError:
        return False
    return rp.can_fetch(user_agent, url)


def polite_get(url: str, delay: float = 2.0, params: dict | None = None, check_robots: bool = True) -> requests.Response:
    """같은 사이트에는 delay초 간격으로만 요청하고, robots.txt가 막은 주소는 받지 않는다."""
    if check_robots and not allowed_by_robots(url):
        raise PermissionError(f"robots.txt가 허용하지 않는 주소입니다: {url}")
    host = urlparse(url).netloc
    wait = _last_request.get(host, 0) + delay - time.time()
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    if resp.status_code == 429:
        time.sleep(min(int(resp.headers.get("Retry-After", "30")), 120))
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request[host] = time.time()
    resp.raise_for_status()
    return resp


def clean_wikitext(wikitext: str) -> str:
    """위키 문법(틀, 링크, 각주, 태그)을 걷어 내고 본문만 남긴다."""
    text = re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", wikitext, flags=re.S)
    for _ in range(5):
        text, n = re.subn(r"\{\{[^{}]*\}\}", "", text)
        if not n:
            break
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_NON_TEXT = [
    ".ws-noexport", "#headertemplate", ".noprint", "sup.reference", "style", ".mw-editsection",
    "table.messagebox", "table.ambox", "table.metadata", ".wst-header", ".pagenum", ".ws-pagenum",
    '[class*="license"]', '[id*="header"]', '[class*="header"]', ".mw-references-wrap", "ol.references",
]
_TRAILING_HEADINGS = re.compile(r"\n+(라이선스|License|授權|许可协议)\s*$")


def html_to_text(html: str) -> str:
    """위키문헌 본문 HTML에서 머리말 틀·라이선스·각주·쪽 번호를 지우고 본문만 남긴다."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for sel in _NON_TEXT:
        for e in soup.select(sel):
            e.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return _TRAILING_HEADINGS.sub("", text)


def fetch_wikisource(title: str, lang: str = "zh") -> str:
    """위키문헌(Wikisource)에서 저작권이 만료된 작품 원문을 받아 정리한다.

    본문이 스캔 쪽에서 끌어오는(<pages>) 문서가 많아, 위키 문법 대신 렌더링된 HTML을 받는다.
    """
    url = f"https://{lang}.wikisource.org/w/api.php"
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": "2", "redirects": "1"}
    data = polite_get(url, params=params, check_robots=False).json()
    if "error" in data:
        raise KeyError(f"위키문헌에서 찾을 수 없습니다: {lang}:{title} ({data['error'].get('info')})")
    return html_to_text(data["parse"]["text"])


def ocr_image(path: str, lang: str = "kor+chi_tra") -> str:
    """Tesseract OCR. Colab에서는 먼저
    !apt-get -qq install tesseract-ocr tesseract-ocr-kor tesseract-ocr-chi-tra
    !pip -q install pytesseract
    를 실행한다. 고문헌은 결과를 반드시 사람이 교정해야 한다."""
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(path), lang=lang)
