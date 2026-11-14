# 『서유기』에서 『구운몽』까지, 데이터로 읽다 — 실습 코드

이 저장소는 『서유기』에서 『구운몽』까지, 데이터로 읽다(부제: 코딩을 몰라도 따라 하는 한중 고전 텍스트마이닝)의 실습 저장소입니다. 지은이는 이현정·최용욱입니다.
책에는 핵심 코드만 싣고, 전체 코드와 실습 노트북은 이곳에서 제공합니다.

책은 두 고전을 끝까지 따라갑니다. 중국의 『서유기』와 한국의 『구운몽』입니다. 두 작품 모두 저작권 보호 기간이 끝난 작품이고, 실습 노트북이 위키문헌(Wikisource)에서 원문을 직접 내려받습니다.

## 바로 실행하기 (Google Colab)

설치 없이 브라우저에서 실행할 수 있습니다. 노트북 위의 **Colab에서 열기** 버튼을 누르고 첫 셀부터 차례로 실행하세요.

| 장 | 노트북 | 다루는 내용 |
|---|---|---|
| 제2장 손오공은 이름이 몇 개일까? | [ch2_corpus](notebooks/ch2_corpus.ipynb) | 번체·간체 정규화, 원문 내려받기, Kiwi·jieba, 호칭 사전과 이름표(메타데이터) |
| 제3장 서유기의 진짜 주인공은 누구일까? | [ch3_lexical](notebooks/ch3_lexical.ipynb) | 인물 빈도, 키워드, 숨은 주제(LDA)와 그 안정성, 문체가 먼 회차(Delta) |
| 제4장 성진은 꿈에서 누가 되었다가 돌아왔을까? | [ch4_meaning](notebooks/ch4_meaning.ipynb) | 꿈이라는 낱말의 분포, word2vec, 다국어 문장 검색, 『장자』 인용 찾기, 감정 점수 |
| 제5장 양소유의 여덟 인연은 누구일까? | [ch5_network](notebooks/ch5_network.ipynb) | 호칭 집계, 같은 줄 공동 출현 관계도, 대표 줄 읽기, 두 작품의 호칭 변화 비교 |
| 제6장 AI가 지어낸 책을 믿어도 될까? | [ch6_llm](notebooks/ch6_llm.ipynb) | 챗봇 질문 문장 만들기, AI가 인용한 구절 검증, 원문-번역 정렬, AI 활용 기록 |

제1장은 관점과 연구 설계를 다루는 장이라 실습 노트북이 없습니다. 제2~5장 노트북에는 실행 결과(표와 그림)가 저장되어 있어 실행하기 전에도 예상 결과를 볼 수 있습니다. 제6장은 API 키 없이 실행되고, 책에 실은 AI 답변(Claude, 2026.11.14.)이 노트북에 들어 있습니다.

- **제2장**을 처음 실행하면 『서유기』 100회를 내려받느라 4~5분 걸립니다. 위키문헌에 부담을 주지 않으려고 요청 사이에 2초씩 쉽니다. 한 번 받으면 `data` 폴더에 저장되어 다시 받지 않습니다. 제3장도 같은 원문을 씁니다.
- **제4장·제6장**은 처음 실행할 때 모델을 내려받아 몇 분 걸립니다. 제4장의 감정 점수 계산도 몇 분 걸립니다.
- **제6장**은 API 키가 필요 없습니다. 노트북이 만들어 주는 질문 문장을 챗봇(Claude, ChatGPT, Gemini 등) 대화창에 붙여 넣어 답을 받고, 그 답을 노트북에 붙여 넣어 검증합니다.

## 내 컴퓨터에서 개발하기

```bash
git clone https://github.com/ITPainter/korea-china-classics-text-mining.git
cd korea-china-classics-text-mining
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

| 명령 | 하는 일 |
|---|---|
| `pip install -e ".[models]"` | 제4장·제6장용 gensim, sentence-transformers 설치 |

## 구성

```
kclit/           공통 모듈
  classics.py    『서유기』·『구운몽』·『장자』 원문 내려받기, 코퍼스 표
  people.py      호칭 사전(손오공, 양소유와 여덟 여인), 호칭 집계
  text.py        정규화, 번체·간체 변환, 문자 체계 판별, 문장 분리
  tokenize.py    Kiwi 형태소 분석, jieba 분사, 고전 한문 글자 단위
  collect.py     robots.txt를 지키는 크롤링, 위키문헌 원문, OCR
  corpus.py      폴더 불러오기, 메타데이터 점검
  freq.py        빈도, 키워드, TF-IDF, 공기어, 낱말 검색
  topics.py      LDA 토픽 모델링
  stylometry.py  Burrows의 Delta, 문체가 먼 회차
  embed.py       word2vec, 다국어 문장 임베딩
  sentiment.py   정서 점수, 정서 곡선, 전환점
  reuse.py       텍스트 재사용·인용 탐지
  network.py     관계도, 중심성
  align.py       원문-번역문 정렬
  llm.py         챗봇 질문 문장, AI 인용 구절 검증, AI 활용 기록
  viz.py         그래프용 한글·한자 글꼴
notebooks/       장별 Colab 노트북
```

## 원문에 대하여

원문은 이 저장소에 들어 있지 않고, 노트북이 위키문헌에서 실행할 때 내려받습니다. 판본과 입력자에 따라 글자가 다를 수 있으니(예: 『장자』는 '胡蝶', 『구운몽』은 '蝴蝶') 분석 전에 정규화하세요.

## 인용

```
이현정·최용욱, 『서유기』에서 『구운몽』까지, 데이터로 읽다: 코딩을 몰라도 따라 하는 한중 고전 텍스트마이닝, 2027.
```

## 라이선스

코드는 [MIT 라이선스](LICENSE)를 따릅니다. 제4장 정서 분석의 모델 구성은 저자의 [sentiment-analysis-multi-language](https://github.com/ITPainter/sentiment-analysis-multi-language)(MIT)에서 가져왔습니다.
