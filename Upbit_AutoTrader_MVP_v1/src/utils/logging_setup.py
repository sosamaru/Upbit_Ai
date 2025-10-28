import logging
from logging.handlers import RotatingFileHandler
import os


def setup_logging(logfile: str = "logs/app.log") -> None:
    """
    콘솔 + 회전 파일 로깅 설정.
    여러 번 호출되더라도 중복 핸들러를 달지 않습니다.
    """
    root = logging.getLogger()
    if root.handlers:
        return  # 이미 설정됨

    root.setLevel(logging.INFO)

    # 콘솔
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s"))
    root.addHandler(ch)

    # 파일(회전)
    try:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s"))
        root.addHandler(fh)
    except Exception:
        # 파일 로거 실패해도 콘솔 로깅은 계속
        pass
