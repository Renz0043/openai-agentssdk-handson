from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import RawResponsesStreamEvent

async def handle_stream_events(stream_result, show_raw_response=True):
    """
    ストリーミングイベントを処理し、適切な出力を行う関数
    
    Args:
        stream_result: ストリーミング結果オブジェクト
        show_raw_response: 生のレスポンスをコンソールに表示するかどうか（デフォルト：True）
    """
    response_text = ""
    async for event in stream_result.stream_events():
        if not isinstance(event, RawResponsesStreamEvent):
            continue
        data = event.data
        if isinstance(data, ResponseTextDeltaEvent):
            if show_raw_response:  # show_raw_responseがTrueの場合のみ表示
                print(data.delta, end="", flush=True)
            response_text += data.delta
        elif isinstance(data, ResponseContentPartDoneEvent):
            if show_raw_response:  # show_raw_responseがTrueの場合のみ表示
                print("\n")
    
    # デバッグ出力を追加
    # print("\n=== final_outputの内容 ===")
    # print("型:", type(stream_result.final_output))
    # print("値:", stream_result.final_output)
    # print("=====================\n")
    
    # 最終的なレスポンスを返す
    return stream_result.final_output