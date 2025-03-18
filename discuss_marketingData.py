import asyncio
import uuid
import pandas as pd
from pandasql import sqldf
from dataclasses import dataclass

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace,Agent, FunctionTool, RunContextWrapper, function_tool

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from agents import set_default_openai_key

set_default_openai_key(OPENAI_API_KEY)

"""
このサンプルは、ハンドオフ/ルーティングパターンを示しています。トリアージエージェントが最初のメッセージを受け取り、
その後、リクエストに基づいて適切なエージェントにハンドオフします。応答はユーザーにストリーミングされます。
"""

@dataclass
class UserInfo:  
    overview: str
    service: str

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
    # DataFrameのto_markdownメソッドを利用（pandas 1.0以降で利用可能）
    return result.to_markdown(index=False)

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
    instructions="ユーザーからの内容に対して、適切なエージェントにハンドオフする",
    handoffs=[dataAnalyst_agent, content_agent, querydata_agent],
)


async def main():
    # この会話のトレースをリンクするために、会話IDを作成します
    conversation_id = str(uuid.uuid4().hex[:16])

    user_info = UserInfo(
        overview="""
        各ページの概要詳細
            •	トップページ: お客様のニーズに応じた情報への動線を明確にし、ブランドの信頼感を高めるビジュアルとキャッチコピーを配置します。主要製品のビジュアルバナー、BtoBサービスの特徴、簡単な企業情報やお知らせへのリンクも設けます。
            •	製品情報（各モデル）ページ:各製品モデルごとに専用ページを用意し、詳細なスペック、使用シーン、ユーザーレビュー、FAQなどを掲載します。各ページ内に問い合わせボタンを設置し、問い合わせフォームに誘導。送信後は「サンクスページ」で確認メッセージを表示し、担当者からの追跡連絡を伝えます。
            •	直接お問い合わせページ:製品に関する質問だけでなく、営業担当と直接話したい場合の問い合わせ専用フォームを用意。企業の信頼性を高めるため、担当者の顔写真や連絡先情報を記載し、フォーム送信後は「サンクスページ」で受付完了を明示します。
            •	企業情報ページ:会社の沿革、理念、組織構成、拠点など、企業としての信頼性や実績を紹介します。また、採用情報やニュースリリースもリンクさせることで、企業の動向に関心を持つユーザーへの情報提供も実施します。
            •	BtoBサービスページ:法人顧客向けに、提供しているサービス内容（例：大量導入サポート、カスタマイズ提案、アフターサポート体制）を詳しく解説。また、実際の導入事例やクライアントの声を掲載することで、信頼性をアピール。問い合わせフォームや資料請求フォームを設け、送信後のサンクスページで受付完了を案内します。
            •	その他補助ページ:FAQでよくある質問に対する回答を提示し、利用規約・プライバシーポリシーで法的情報を明確にするなど、ユーザーの安心感と信頼感を高めるページを追加します。
        """,
        service="""
        株式会社テストパソコンは、ユーザーのニーズに合わせた幅広いPCラインナップと充実したサービスを提供しています。
        •	PCの品揃え
            •	ハイグレードモデルのPC:高性能で最新技術を搭載したモデル。クリエイティブ作業や高負荷な業務に適しており、スペック重視のプロフェッショナル向けです。
            •	ミドルモデルのPC:性能とコストパフォーマンスのバランスが取れたモデル。一般ユーザーや中小企業のオフィス用途など、幅広いシーンで活用できる製品ラインです。
            •	エントリーモデルのPC:手頃な価格で基本機能をしっかりカバー。初めてPCを導入する方や予算を抑えたいユーザー向けに最適です。
        •	提供しているサービス
            •	各製品の専用問い合わせ:各モデルの詳細ページ内に問い合わせフォームを設置し、製品に関する疑問や詳細な相談ができる仕組みを用意。問い合わせ後は専用のサンクスページで受付完了を案内。
            •	直接営業担当へのお問い合わせ:より詳しい製品説明や提案を希望するお客様向けに、営業担当と直接相談できる専用ページを提供。
            •	BtoBサービス:法人向けに、大量導入やカスタマイズの提案、アフターサポート体制など、企業専用のサービスを展開。導入事例やサービス内容の詳細情報を提供し、専用の問い合わせ窓口も設けています。
            •	企業情報およびサポート:企業概要、沿革、採用情報など、会社としての信頼性を示す情報や、FAQ、プライバシーポリシー、利用規約といったサポート関連の情報を掲載し、安心して利用できる環境を整えています。

        このように、株式会社テストパソコンは、個人から法人まで幅広い顧客層に向けた多様な製品と、直接の問い合わせ、法人向け専用サービスなど、顧客のニーズに合わせた柔軟なサポート体制を提供しています。
        """
    )

    msg = input("今日はBtoBマーケティングの何についてお話しますか？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        # 各会話のターンは単一のトレースとなります。通常、ユーザーからの各入力は
        # あなたのアプリへのAPIリクエストとなり、それをtrace()でラップすることができます
        with trace("Routing example", group_id=conversation_id):
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