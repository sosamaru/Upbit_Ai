"""
업비트 웹소켓 구독(선택)
- SIMPLE ticker 스트림 구독 예시
"""

from __future__ import annotations
import json
import threading
from websocket import WebSocketApp

UPBIT_WS = "wss://api.upbit.com/websocket/v1"


def subscribe_ticker(tickers: list[str], on_message):
    """
    tickers 예: ["KRW-BTC", "KRW-ETH"]
    on_message: 콜백(bytes)->None
    """
    def _on_open(ws):
        sub = [
            {"ticket": "autotrader"},
            {"type": "ticker", "codes": tickers, "isOnlyRealtime": True},
            {"format": "SIMPLE"},
        ]
        ws.send(json.dumps(sub))

    def _on_msg(ws, msg):
        try:
            on_message(msg)
        except Exception:
            pass

    ws = WebSocketApp(UPBIT_WS, on_open=_on_open, on_message=_on_msg)
    th = threading.Thread(target=ws.run_forever, daemon=True)
    th.start()
    return ws, th
