import asyncio
import uuid
import pandas as pd
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

class BusinessTopicOutput(BaseModel):
    is_business_topic: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="BtoBマーケティングにおけるデータ分析またはSEOに関するノウハウ以外を質問されていないかチェックする。",
    output_type=BusinessTopicOutput,
)

@input_guardrail
async def businessTopic_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_business_topic,
    )

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

async def main():
    user_info = UserInfo(
        site_id = "111"
    )

    msg = input("今日はBtoBマーケティングの何についてお話しますか？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
            # 各会話のターンは単一のトレースとなります。通常、ユーザーからの各入力は
            # あなたのアプリへのAPIリクエストとなり、それをtrace()でラップすることができます
            with trace("Routing example"):
                result = Runner.run_streamed(
                    agent,
                    input=inputs,
                    context=user_info,
                )
                async for event in result.stream_events():
                    if not isinstance(event, RawResponsesStreamEvent):
                        continue
                    data = event.data
                    if isinstance(data, ResponseTextDeltaEvent):
                        print(data.delta, end="", flush=True)
                    elif isinstance(data, ResponseContentPartDoneEvent):
                        print("\n")

            inputs = result.to_input_list()
            print("\n")

            user_msg = input("更に話したいことがあれば教えて下さい: ")
            inputs.append({"content": user_msg, "role": "user"})
            # 都度トリアージに設定し直す
            #agent = result.current_agent
            agent = triage_agent


if __name__ == "__main__":
    asyncio.run(main())