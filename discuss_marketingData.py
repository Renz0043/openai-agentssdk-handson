import asyncio
import uuid
import pandas as pd
import json
from pandasql import sqldf
from dataclasses import dataclass

from pydantic import BaseModel, Field

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import (
    Agent,
    FunctionTool,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RawResponsesStreamEvent,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    function_tool,
    input_guardrail,
    trace
)

import os
from dotenv import load_dotenv
from stream_handler import handle_stream_events

# .envファイルを読み込む
load_dotenv()

# 環境変数からAPIキーを取得
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# デフォルトキーとして登録
from agents import set_default_openai_key
set_default_openai_key(OPENAI_API_KEY)

"""
このサンプルは、ハンドオフ/ルーティングパターンを示しています。トリアージエージェントが最初のメッセージを受け取り、
その後、リクエストに基づいて適切なエージェントにハンドオフします。応答はユーザーにストリーミングされます。
"""

@dataclass
class UserInfo:  
    site_id: str = Field(description="サイトを特定する一意の文字列")

@function_tool
async def query_data(date_range_from: str, date_range_to: str) -> str:
    """
    引数で渡された期間におけるデータを抽出するツール。
    Args: 
        data_range_from: 抽出対象となるデータの開始日。yyyy-mm-dd形式で渡される。
        data_range_to: 抽出対象となるデータの終了日。yyyy-mm-dd形式で渡される。
    """

    df = pd.read_csv('landingpage_data.csv')

    # 日付でクエリを実行（例として '2025/01/01' で絞り込み）
    query = f"SELECT * FROM df WHERE date BETWEEN '{date_range_from}' AND '{date_range_to}'"
    result = sqldf(query, locals())

    # 結果をマークダウン形式で出力
    return result.to_markdown(index=False)

@function_tool
async def query_service(wrapper: RunContextWrapper[UserInfo]) -> str:
    """
    引数で渡されたコンテキストをもとに、サイト情報を抽出するツール。
    Args: 
        site_id: サイトを特定する一意の文字列。
    """

    df = pd.read_csv('site_data.csv')

    # 日付でクエリを実行（例として '2025/01/01' で絞り込み）
    query = f"SELECT service, overview FROM df WHERE site_id = {wrapper.context.site_id}"
    result = sqldf(query, locals())

    # 結果をマークダウン形式で出力
    return result.to_markdown(index=False)

# エージェントの定義
dataAnalyst_agent = Agent(
    model="gpt-4o-mini",
    name="dataAnalyst_agent",
    instructions="あなたは優秀なデータアナリストです。特にBtoBマーケティングにおけるWebサイト運用データの分析に深い知見があります。",
)

content_agent = Agent(
    model="gpt-4o-mini",
    name="content_agent",
    instructions="あなたは優秀なSEOコンテンツの作成担当です。特にBtoBマーケティングに効果的なコンテンツを作成するための深い知見があります。",
)

querydata_agent = Agent(
    model="gpt-4o-mini",
    name="querydata_agent",
    instructions="あなたは優秀なインフラエンジニアです。自社が運用しているwebサイトのデータをクエリ、加工した結果の表をマークダウン形式ですべて省略せずに出力します。ユーザーの入力からクエリ対象期間の開始日と終了日をyyyy-mm-dd形式で抽出してツールを利用します。",
    tools=[query_data],
    handoffs=[dataAnalyst_agent, content_agent],
)

class OutputIdentifyDate(BaseModel):
    is_date: bool
    date_from: str
    date_to: str
    reasoning: str

identify_date_agent = Agent(
    model="gpt-4o-mini",
    name="identify_date_agent",
    instructions="""
    # 役割
    ユーザーから、抽出するデータの対象期間を確実にヒアリングする。

    # 解釈
    - ユーザーは基本的に、最も新しい年のデータを要求するが、指定があればこの限りではない。
    - ユーザーの入力にはゆらぎが発生することを考慮する。以下には2月を指定する例を示す。:[2月, 02, feb]
    - ユーザーの入力を下に、ヒアリングすべき内容に落とし込んで特定できない場合、ユーザーが意図している内容をreasoningに含めてユーザーに確認する。
    
    ## ヒアリングすべき内容
    - クエリの開始日。yyyy-mm-dd形式。
    - クエリの終了日。yyyy-mm-dd形式。
    """,
    output_type=OutputIdentifyDate
)

triage_agent = Agent(
    model="gpt-4o-mini",
    name="triage_agent",
    instructions="""
    # 役割
    ユーザーからの内容に対して、適切なエージェントにハンドオフする。

    # ハンドオフポリシー
    ## ハンドオフせずに、自身で実行すべきタスク
    - site_idにもとづくサービス情報のクエリ
    ## ハンドオフすべきタスクとそのターゲット
    - データの抽出が必要な場合はquerydata_agentへハンドオフする。この場合は抽出対象となるデータの開始日と終了日がyyyy-mm-dd形式で判別できるまでユーザーに質問する。
    - SEOコンテンツに関する話題ははcontent_agentへハンドオフする。
    - データの分析やレポートティングはdataAnalyst_agentへハンドオフする。

    """,
    tools=[query_service],
    handoffs=[dataAnalyst_agent, content_agent, querydata_agent],
)




## クエリ日付特定フェーズ -----------------------
async def identifyQueryDate(inputs: list[TResponseInputItem]) -> dict:
    result = Runner.run_streamed(
        identify_date_agent,
        inputs,
    )
    response = await handle_stream_events(result)
    
    if response is None:
        raise ValueError("レスポンスが取得できませんでした")

    # responseの型に応じて適切に処理
    if isinstance(response, str):
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("JSONのパースに失敗しました")
    else:
        response_json = response

    
    if response_json.is_date:
        # date_fromとdate_toの値を確認
        if response_json.date_from and response_json.date_to:
            print(f"推定された期間: {response_json.date_from} から {response_json.date_to}")
            user_confirm = input("この期間でよろしいですか？ (y/n): ")
            if user_confirm.lower() == 'y':
                print("日付の推定に成功しました。")
                return response_json
            else:
                print("日付の推定に失敗しました。")
                print(response_json.reasoning)
                new_input = input("日付を正しい形式で入力してください（例：2024-02-01）: ")
                inputs.append({"content": new_input, "role": "user"})
                return await identifyQueryDate(inputs)
        else:
            print("日付の推定に失敗しました。")
            print(response_json.reasoning)
            new_input = input("日付を正しい形式で入力してください（例：2024-02-01）: ")
            inputs.append({"content": new_input, "role": "user"})
            return await identifyQueryDate(inputs)
    else:
        print("日付の推定に失敗しました。")
        print(response_json.reasoning)
        new_input = input("日付を正しい形式で入力してください（例：2024-02-01）: ")
        inputs.append({"content": new_input, "role": "user"})
        return await identifyQueryDate(inputs)

## -----------------------------------------

## サイト情報抽出フェーズ -----------------------

## -----------------------------------------

## データ抽出フェーズ -----------------------

## -----------------------------------------

## レポーティングフェーズ -----------------------

## -----------------------------------------




async def main():
    conversation_id = str(uuid.uuid4().hex[:16])

    user_info = UserInfo(
        site_id="111"
    )

    msg = input("今日はBtoBマーケティングの何についてお話しますか？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    try:
        with trace("Marketing Discussion"):
            ## クエリ日付特定フェーズ -----------------------
            try:
                query_result = await identifyQueryDate(inputs)
                # デバッグ出力
                print(f"クエリ日付特定フェーズの結果: {query_result}")
                date_from = query_result.date_from
                date_to = query_result.date_to

            except ValueError as e:
                print(f"クエリ日付特定時にエラーが発生しました: {str(e)}")
            ## -----------------------------------------

            ## サイト情報抽出フェーズ -----------------------
            
            ## -----------------------------------------

            ## データ抽出フェーズ -----------------------

            ## -----------------------------------------

            ## レポーティングフェーズ -----------------------
            
            ## -----------------------------------------

    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {str(e)}")
        print("もう一度お試しください。")


if __name__ == "__main__":
    asyncio.run(main())


