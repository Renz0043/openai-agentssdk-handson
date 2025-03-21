from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import RawResponsesStreamEvent

async def handle_stream_events(stream_result):
    """
    ストリーミングイベントを処理し、適切な出力を行う関数
    
    Args:
        stream_result: ストリーミング結果オブジェクト
    """
    async for event in stream_result.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            print(data.delta, end="", flush=True)
        elif isinstance(data, ResponseContentPartDoneEvent):
            print("\n") 