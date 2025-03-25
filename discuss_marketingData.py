# 標準ライブラリ
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List

# サードパーティライブラリ
import pandas as pd
from dotenv import load_dotenv
from pandasql import sqldf
from pydantic import BaseModel, Field

# OpenAI関連
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
    set_default_openai_key,
    trace
)
# 自作モジュール
from stream_handler import handle_stream_events

# .envファイルを読み込む
load_dotenv()
# 環境変数からAPIキーを取得
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# デフォルトキーとして登録
set_default_openai_key(OPENAI_API_KEY)

"""
このサンプルは、ハンドオフ/ルーティングパターンを示しています。トリアージエージェントが最初のメッセージを受け取り、
その後、リクエストに基づいて適切なエージェントにハンドオフします。応答はユーザーにストリーミングされます。

主な機能:
- UserInfoクラス: ユーザー情報を保持するデータクラス
- query_data: 指定された期間のデータを抽出するツール
- query_service: サイト情報を抽出するツール

使用方法:
1. 必要な環境変数(.env)を設定
2. 必要なCSVファイル(landingpage_data.csv, site_data.csv)を配置
3. エージェントを実行してデータ分析を開始

依存関係:
- Python 3.7+
- OpenAI API
- pandas
- pandasql
- pydantic
- python-dotenv

"""

## ユーザー情報クラス ----------------------------
@dataclass
class UserInfo:  
    execution_date: str = Field(description="実行日")
    site_id: str = Field(description="サイトを特定する一意の文字列")
    service_info: str = Field(default="", description="サイトのサービス情報")
    access_data: str = Field(default="", description="アクセスデータ")
    report_result: str = Field(default="", description="レポーティングデータ")

## データ抽出ツール ----------------------------
@function_tool
async def query_data(date_range_from: str, date_range_to: str, columns: List[str], groupby_columns: List[str] = None) -> str:
    """
    引数で渡された期間におけるデータを抽出するツール。
    
    Args:
        columns: List[str] 抽出対象となるカラム名がリスト形式で渡される。
        date_range_from: 抽出対象となるデータの開始日。yyyy-mm-dd形式で渡される。
        date_range_to: 抽出対象となるデータの終了日。yyyy-mm-dd形式で渡される。 
        groupby_columns: List[str] グループ化対象となるカラム名がリスト形式で渡される。省略可能。

    Returns:
        str: マークダウン形式のデータ

    Raises:
        FileNotFoundError: CSVファイルが見つからない場合
        ValueError: 日付形式が不正な場合
    """
    try:
        # 日付形式の検証
        datetime.strptime(date_range_from, '%Y-%m-%d')
        datetime.strptime(date_range_to, '%Y-%m-%d')
        
        # カラムのバリデーション
        if not columns:
            raise ValueError("抽出対象のカラムを指定してください")

        # データを読み込む
        df = pd.read_csv('landingpage_data.csv')
        
        # カラムをカンマ区切りの文字列に変換
        columns_str = ", ".join(columns)

        # 基本のクエリを構築
        query = f"SELECT {columns_str} FROM df WHERE date BETWEEN '{date_range_from}' AND '{date_range_to}'"
        
        # groupby_columnsが指定されている場合、GROUP BY句を追加
        if groupby_columns:
            groupby_str = ", ".join(groupby_columns)
            query += f" GROUP BY {groupby_str}"

        # クエリ実行
        result = sqldf(query, locals())

        if result.empty:
            return "指定された条件に一致するデータが見つかりませんでした。"

        # 結果をマークダウン形式で出力
        return result.to_markdown(index=False)

    except FileNotFoundError:
        return "データファイル(landingpage_data.csv)が見つかりません。"
    except ValueError as e:
        return f"入力値が不正です: {str(e)}"
    except Exception as e:
        return f"予期せぬエラーが発生しました: {str(e)}"

## サイト情報抽出ツール ----------------------------
@function_tool
async def query_service(site_id: str) -> str:
    """
    引数で渡されたコンテキストをもとに、サイト情報を抽出するツール。

    Args:
        site_id: サイトを特定する一意の文字列。

    Returns:
        str: サイト情報をマークダウン形式で返却

    Raises:
        FileNotFoundError: CSVファイルが見つからない場合
        ValueError: site_idが不正な場合
    """
    try:
        # site_idのバリデーション
        if not site_id:
            raise ValueError("site_idを指定してください")

        # データを読み込む
        df = pd.read_csv('site_data.csv')

        # シンプルなクエリを実行
        query = f"SELECT service, overview FROM df WHERE site_id = '{site_id}';"
        result = sqldf(query, locals())

        if result.empty:
            return "指定されたsite_idに一致するデータが見つかりませんでした。"

        # 結果をマークダウン形式で出力
        return result.to_markdown(index=False)

    except FileNotFoundError:
        return "データファイル(site_data.csv)が見つかりません。"
    except ValueError as e:
        return f"入力値が不正です: {str(e)}"
    except Exception as e:
        return f"予期せぬエラーが発生しました: {str(e)}"

## エージェントの定義 ----------------------------
dataAnalyst_agent = Agent(
    model="gpt-4o-mini",
    name="dataAnalyst_agent",
    instructions="""
    # 役割と責任
    あなたは優秀なデータアナリストとして、以下の責務を担います：
    - Webサイトの運用データを分析し、実用的なインサイトを導き出す
    - KPIの推移や相関関係を分析し、改善提案を行う
    - データに基づいた意思決定をサポートする

    # 専門知識
    - BtoBマーケティングにおけるWebサイト運用データ分析
    - コンバージョン最適化
    - ユーザー行動分析
    - アクセス解析

    # コミュニケーション
    - 技術的な用語は必要に応じて平易な言葉で説明
    - 結論から述べ、詳細は後述する構成
    - データの解釈と実践的な提案を明確に区別
    - 分析の限界や前提条件を適切に説明
    """,
)

content_agent = Agent(
    model="gpt-4o-mini",
    name="content_agent",
    instructions="""
    # 役割と責任
    あなたは優秀なSEOコンテンツの作成担当として、以下の責務を担います：
    - BtoBマーケティングに特化したSEOコンテンツの企画・作成
    - ターゲットとなるペルソナに合わせたコンテンツ設計
    - 検索意図を考慮したキーワード選定と最適化
    - コンバージョンを意識した記事構成の提案

    # 専門知識
    - BtoB業界特有のキーワード戦略
    - ロングテール・ショートテールキーワードの使い分け
    - コンテンツマーケティングの最新トレンド
    - SEOテクニカル要件の理解

    # コミュニケーション方針
    - 提案は具体的な数値目標と共に提示
    - 業界用語は必要に応じて解説を付記
    - 競合分析に基づく差別化ポイントの明確化
    - コンテンツ改善のPDCAサイクルを意識した提案
    """,
)

querydata_agent = Agent(
    model="gpt-4o-mini",
    name="querydata_agent",
    instructions="""
    # 定義
    あなたは優秀なインフラエンジニアです。

    # 役割
    自社が運用しているwebサイトのデータをクエリ文を作成して、抽出結果をマークダウン形式ですべて省略せずに出力します。
    ユーザーの入力からクエリ対象期間の開始日と終了日(yyyy-mm-dd形式)、対象のカラムを識別してツールを利用します。

    # 留意点
    ユーザーからの入力にはゆらぎが発生することを考慮し、すでに存在するカラム名と照合して適切と思われるカラム名を選択してください。
    
    # テーブル
    | カラム名         | 日本語名             | データ型         | 説明 |
    |------------------|----------------------|------------------|------|
    | `date`           | 日付                 | DATE             | データの記録日（例：2025-01-03） |
    | `ページタイトル` | ページタイトル       | STRING           | Webページのタイトル（例：トップページ） |
    | `URL`            | URL                  | STRING           | 対象ページのURL |
    | `訪問数`         | 訪問数               | INTEGER          | 対象ページへの訪問回数 |
    | `直帰率`         | 直帰率               | STRING (PERCENT) | ページからすぐに離脱したユーザーの割合（例：35%） |
    | `平均滞在時間`   | 平均滞在時間         | TIME (hh:mm:ss)  | ページに滞在した平均時間（例：0:02:45） |
    | `CV数`           | コンバージョン数     | INTEGER          | コンバージョンに至った件数 |
    | `CV率`           | コンバージョン率     | STRING (PERCENT) | コンバージョン率（例：1.67%） |

    # 元のクエリ
    -- クエリ1: コンバージョン率が高いページTOP5
    -- ページのコンバージョン率（CV率）が高い順に並べ替え、上位5件を取得します。
    -- クエリ開始
    SELECT *
    FROM df
    ORDER BY CAST(SUBSTR(CV率, 1, LENGTH(CV率) - 1) AS REAL) DESC
    LIMIT 5;
    -- クエリ終了

    -- クエリ2: 平均滞在時間が長いページTOP5
    -- ユーザーの平均滞在時間が長いページを上位5件取得します。
    -- クエリ開始
    SELECT *
    FROM df
    ORDER BY
    CAST(SUBSTR(平均滞在時間, 1, INSTR(平均滞在時間, ':') - 1) AS INTEGER) * 3600 +
    CAST(SUBSTR(平均滞在時間, INSTR(平均滞在時間, ':') + 1, INSTR(平均滞在時間, ':', -1) - INSTR(平均滞在時間, ':') - 1) AS INTEGER) * 60 +
    CAST(SUBSTR(平均滞在時間, INSTR(平均滞在時間, ':', -1) + 1) AS INTEGER) DESC
    LIMIT 5;
    -- クエリ終了

    -- クエリ3: 訪問数が多いページTOP5（人気ページの特定）
    -- 訪問数が多いページを上位5件取得し、人気のあるページを特定します。
    -- クエリ開始
    SELECT *
    FROM df
    ORDER BY 訪問数 DESC
    LIMIT 5;
    -- クエリ終了

    -- クエリ4: ページ単位のCV率・訪問数・直帰率を集計（日付なし）
    -- ページごとに訪問数の合計、直帰率の平均、CV率の平均を集計します。
    -- クエリ開始
    SELECT ページタイトル,
        SUM(訪問数) AS 訪問数合計,
        AVG(CAST(SUBSTR(直帰率, 1, LENGTH(直帰率) - 1) AS REAL)) AS 平均直帰率,
        AVG(CAST(SUBSTR(CV率, 1, LENGTH(CV率) - 1) AS REAL)) AS 平均CV率
    FROM df
    GROUP BY ページタイトル;
    -- クエリ終了

    -- クエリ5: 日別CV数の推移
    -- 日ごとのコンバージョン数（CV数）の推移を確認します。
    -- クエリ開始
    SELECT date, SUM(CV数) AS 日別CV数
    FROM df
    GROUP BY date
    ORDER BY date;
    -- クエリ終了

    -- クエリ6: CV数と訪問数の相関チェック（散布図用）
    -- 訪問数とCV数の関係性を確認するためのデータを取得します。
    -- クエリ開始
    SELECT 訪問数, CV数 FROM df;
    -- クエリ終了

    -- クエリ7: CV率が平均より高いページの抽出
    -- 全体の平均CV率より高いページを抽出します。
    -- クエリ開始
    SELECT *
    FROM df
    WHERE CAST(SUBSTR(CV率, 1, LENGTH(CV率) - 1) AS REAL) >
        (SELECT AVG(CAST(SUBSTR(CV率, 1, LENGTH(CV率) - 1) AS REAL)) FROM df);
    -- クエリ終了

    -- クエリ8: 直帰率が高すぎるページの特定（例：50%以上）
    -- 直帰率が50%以上のページを特定し、改善の必要があるページを洗い出します。
    -- クエリ開始
    SELECT *
    FROM df
    WHERE CAST(SUBSTR(直帰率, 1, LENGTH(直帰率) - 1) AS REAL) >= 50;
    -- クエリ終了

    -- クエリ9: ページごとのCV数合計と訪問数合計
    -- ページごとに訪問数とCV数の合計を集計し、CV数の多い順に並べ替えます。
    -- クエリ開始
    SELECT ページタイトル, SUM(訪問数) AS 訪問数合計, SUM(CV数) AS CV数合計
    FROM df
    GROUP BY ページタイトル
    ORDER BY CV数合計 DESC;
    -- クエリ終了

    -- クエリ10: ページごとの平均滞在時間が長い順（人気・エンゲージメント測定）
    -- ページごとの平均滞在時間を秒単位で計算し、長い順に並べ替えます。
    -- クエリ開始
    SELECT ページタイトル,
        AVG(
            CAST(SUBSTR(平均滞在時間, 1, INSTR(平均滞在時間, ':') - 1) AS INTEGER) * 3600 +
            CAST(SUBSTR(平均滞在時間, INSTR(平均滞在時間, ':') + 1, INSTR(平均滞在時間, ':', -1) - INSTR(平均滞在時間, ':') - 1) AS INTEGER) * 60 +
            CAST(SUBSTR(平均滞在時間, INSTR(平均滞在時間, ':', -1) + 1) AS INTEGER)
        ) AS 平均滞在時間秒
    FROM df
    GROUP BY ページタイトル
    ORDER BY 平均滞在時間秒 DESC;
    -- クエリ終了

    # 出力フォーマット:
    ### 抽出結果
    [抽出結果を記載]

    ### 使用したクエリ
    [使用したクエリを記載]
    
    """,
    tools=[query_data],
    handoffs=[dataAnalyst_agent, content_agent],
)

class OutputIdentifyDate(BaseModel):
    date_from: str = Field(description="クエリの開始日（yyyy-mm-dd形式）")
    date_to: str = Field(description="クエリの終了日（yyyy-mm-dd形式）") 
    reasoning: str = Field(description="日付特定の理由付けや、特定できない場合のユーザーへの確認事項")

identify_date_agent = Agent(
    model="gpt-4o-mini",
    name="identify_date_agent",
    instructions="""
    # 役割
    ユーザーから、抽出するデータの対象期間を確実にヒアリングする。

    # 解釈
    - ユーザーは基本的に会話履歴における現在の年度のデータを要求するが、指定があればこの限りではない。
    - 会話履歴に含まれている年度よりも新しいデータを要求することはない。
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
    # 役割と責任
    - ユーザーの要求を正確に理解し、最適なエージェントへ振り分ける
    - 必要に応じて追加情報をユーザーに確認する
    - 複数エージェントの連携が必要な場合の調整を行う

    # 入力の解釈とルーティング基準
    1. データ分析要求の判定
        - 数値分析、トレンド分析 → dataAnalyst_agent
        - 改善提案、インサイト抽出を含む → dataAnalyst_agent
        - KPI、目標達成状況の確認 → dataAnalyst_agent

    2. データ抽出要求の判定
        - 生データの取得 → querydata_agent
        - 期間指定データの抽出 → querydata_agent
        - カスタムクエリの実行 → querydata_agent
        ※ 日付形式(yyyy-mm-dd)の確認を徹底

    3. コンテンツ関連要求の判定
        - SEO施策の提案 → content_agent
        - コンテンツ改善案 → content_agent
        - キーワード戦略 → content_agent

    4. サービス情報要求の判定
        - site_idベースの情報取得 → 自身で処理

    # 品質管理基準
    1. 要求の明確化
        - 期待する成果物のフォーマットを明確にする

    2. エラー防止
        - 日付形式のバリデーション
        - データ範囲の妥当性チェック

    3. 効率的なハンドリング
        - 複数要求の適切な分解
        - 並行処理可能なタスクの識別
        - 依存関係のあるタスクの順序付け

    # コミュニケーション方針
    - 明確で簡潔な確認
    - 専門用語の適切な言い換え
    """,
    tools=[query_service],
    handoffs=[dataAnalyst_agent, content_agent, querydata_agent],
)




## クエリ日付特定フェーズ -----------------------
async def identifyQueryDate(inputs: list[TResponseInputItem], wrapper: RunContextWrapper[UserInfo]) -> OutputIdentifyDate:
    """日付の特定を行う関数
    
    Args:
        inputs: ユーザーとの対話履歴
        wrapper: 実行コンテキスト情報
        
    Returns:
        OutputIdentifyDate: 日付特定結果
        
    Raises:
        ValueError: レスポンスの取得やパースに失敗した場合
    """
    try:
        # エージェントを実行
        result = Runner.run_streamed(identify_date_agent, inputs)
        response = await handle_stream_events(result, show_raw_response=False)  # 生のレスポンスを表示しないようにする
        
        if not response:
            raise ValueError("> System: レスポンスが取得できませんでした")

        # レスポンスをパース
        response_json = (
            json.loads(response) if isinstance(response, str) else response
        )
        
        # 日付が特定できた場合
        if response_json.date_from and response_json.date_to:
            print(f"> AI: 推定された期間: {response_json.date_from} から {response_json.date_to}")
            
            if input("> AI: この期間でよろしいですか？ (y/n): ").lower() == 'y':
                print("> AI: 日付の推定に成功しました。利用可能なカラムをロードしています...")
                return response_json
                
        # 日付の再入力を要求
        print("> AI: 日付の推定に失敗しました。")            
        new_input = input("> AI: 再度分析期間を入力してください（例：2025-02-01から2025-02-28）: ")
        inputs.append({"content": new_input, "role": "user"})
        return await identifyQueryDate(inputs, wrapper)
        
    except json.JSONDecodeError:
        raise ValueError("> System: JSONのパースに失敗しました") 
    except Exception as e:
        raise ValueError(f"> System: 予期せぬエラーが発生しました: {str(e)}")

## -----------------------------------------

## サイト情報抽出フェーズ -----------------------
async def QueryServiceInfo(inputs: list[TResponseInputItem], wrapper: RunContextWrapper[UserInfo]) -> dict:
    """サイト情報を抽出する関数

    Args:
        inputs (list[TResponseInputItem]): ユーザーとの対話履歴
        wrapper (RunContextWrapper[UserInfo]): 実行コンテキスト情報

    Returns:
        dict: サイト情報の抽出結果

    Raises:
        ValueError: レスポンスの取得に失敗した場合
    """
    try:
        # サイト情報を抽出する指示を作成
        query_service_instruction = f"""
        # 指示
        site_idにもとづくサービス情報のクエリをお願いします。

        # site_id
        site_id: {wrapper.site_id}

        # 留意事項
        - 出力は、サービス情報のみを回答することとし、site_idに関する内容は回答に含めないこととします。
        """

        # サイト情報を抽出する指示をinputsに追加
        inputs.append({"content": query_service_instruction, "role": "user"})

        # エージェントを実行してレスポンスを取得
        response = Runner.run_streamed(
            triage_agent,
            inputs,  # 対話履歴を含めて文脈を維持
        )
        result = await handle_stream_events(response, show_raw_response=False)  # 生のレスポンスを表示しないように設定

        if result is None:
            raise ValueError("サイト情報の取得に失敗しました")

        return result

    except Exception as e:
        raise ValueError(f"サイト情報の抽出中にエラーが発生しました: {str(e)}")
## -----------------------------------------

## アクセスデータ抽出フェーズ -----------------------
class OutputAccessData(BaseModel):
    columns: List[str] = Field(description="抽出対象のカラム名リスト")
    reasoning: str = Field(description="カラム特定の理由付けや、特定できない場合のユーザーへの確認事項")

async def QueryAccessData(inputs: list[TResponseInputItem], wrapper: RunContextWrapper[UserInfo], date_from: str, date_to: str, request_text: str) -> OutputAccessData:
    """アクセスデータを抽出する関数

    Args:
        inputs (list[TResponseInputItem]): ユーザーとの対話履歴
        wrapper (RunContextWrapper[UserInfo]): 実行コンテキスト情報
        date_from (str): 抽出開始日
        date_to (str): 抽出終了日
        request_text (str): ユーザーからのリクエスト内容

    Returns:
        OutputAccessData: 抽出されたアクセスデータ情報

    Raises:
        ValueError: レスポンスの取得に失敗した場合
    """
    try:
        # カラム特定エージェントを定義
        identify_columns_agent = Agent(
            model="gpt-4o-mini",
            name="identify_columns_agent",
            instructions="""
            # 役割
            あなたはWebデータ分析のためのカラム特定専門家です。
            ユーザーの入力からデータ抽出対象のカラム名のみを識別してください。

            # カラム特定の詳細ルール
            1. 以下のリストにあるカラム名のみを返してください
            2. リストにないカラム名は絶対に返さないでください
            3. 入力にマッチするカラムが見つからない場合は、空のリストを返してください
            4. 「PC品揃え」などの製品情報や、「提供サービス」などのサービス情報はデータカラムではありません
            
            # 利用可能なカラム（この中からのみ選択）
            - date：日付データ
            - ページタイトル：Webページのタイトル
            - URL：対象ページのURL
            - 訪問数：訪問回数（別名：PV、ページビュー、アクセス数）
            - 直帰率：離脱率（別名：バウンス率）
            - 平均滞在時間：ユーザーの滞在時間
            - CV数：コンバージョン数（別名：成約数）
            - CV率：コンバージョン率（別名：成約率）

            # 入力パターンの解釈
            - スペース、カンマ「,」、読点「、」、接続詞「と」などで複数のカラムが区切られていることがあります
            - 「PV」「コンバージョン」など、略語や別名でカラムが指定されていることがあります
            - 「すべて」「全部」などの指定があれば、すべてのカラムを返します
            - 「SS数」など不明なカラム名は無視してください
            
            # 出力形式
            columns: 特定されたカラム名のリスト（例：["訪問数", "CV数"]）
            reasoning: カラム特定の理由や、不明点がある場合の説明
            """,
            output_type=OutputAccessData
        )
        
        # カラム特定を実行（会話履歴ではなく、単一の指示として渡す）
        column_input = [{"role": "user", "content": f"次のテキストからデータカラムを特定してください: {request_text}"}]
        
        result = Runner.run_streamed(identify_columns_agent, column_input)
        columns_response = await handle_stream_events(result, show_raw_response=False)  # 生のレスポンスを表示しないように設定
        
        if not columns_response:
            raise ValueError("カラム特定に失敗しました")
            
        # レスポンスをパース
        columns_json = (
            json.loads(columns_response) if isinstance(columns_response, str) else columns_response
        )
        
        # 特定されたカラムを表示して確認
        if hasattr(columns_json, 'columns'):
            # 利用可能なカラムのリスト
            available_columns = [
                'date', 'ページタイトル', 'URL', '訪問数', 
                '直帰率', '平均滞在時間', 'CV数', 'CV率'
            ]
            
            # 不正なカラムがある場合はフィルタリング
            valid_columns = [col for col in columns_json.columns if col in available_columns]
            
            # カラムがない場合は、単純にマニュアル判定を試みる
            if not valid_columns:
                # 簡易的なキーワード判定で再試行
                keywords_map = {
                    'date': ['日付', '日'],
                    'ページタイトル': ['タイトル', 'title', 'ページ名'],
                    'URL': ['url', 'リンク', 'アドレス'],
                    '訪問数': ['訪問', 'visit', 'pv', 'ページビュー', 'アクセス'],
                    '直帰率': ['直帰', 'bounce', 'バウンス'],
                    '平均滞在時間': ['滞在', 'time', '時間'],
                    'CV数': ['cv数', 'コンバージョン数', 'conversion', 'コンバージョン', '成約'],
                    'CV率': ['cv率', 'コンバージョン率', 'cvr', '成約率']
                }
                
                # 入力テキストを小文字に変換して空白を削除
                clean_text = request_text.lower().strip()
                
                for column, keywords in keywords_map.items():
                    for keyword in keywords:
                        if keyword.lower() in clean_text and column not in valid_columns:
                            valid_columns.append(column)
                
                # それでもカラムが特定できない場合
                if not valid_columns:
                    print("\n有効なカラムが特定できませんでした。")
                    print("以下のカラムのみ利用可能です:")
                    for col in available_columns:
                        print(f"- {col}")
                    new_input = input("分析したいデータ項目を再入力してください（例: 日付,訪問数,CV率）: ")
                    return await QueryAccessData(inputs, wrapper, date_from, date_to, new_input)
            
            # 推定されたカラムを表示
            print(f"\n推定されたカラム: {', '.join(valid_columns)}")
            
            if input("これらのカラムで抽出を行いますか？ (y/n): ").lower() == 'y':
                print("カラムの特定に成功しました。データを抽出して、レポートを生成します。")
                
                # カラムが確定したので実際にデータを抽出
                data_extraction_instruction = f"""
                # 指示
                データを抽出をお願いします。また抽出時に利用したクエリに併記して。
                抽出対象のサイトは、{wrapper.site_id} です。
                抽出対象の期間は、{date_from} から {date_to} です。

                # 抽出カラム
                {', '.join(valid_columns)}
                
                # 留意事項
                - 出力は基本的にディメンションの値でグループ化することとします。
                - ディメンションの値は、date、ページタイトル、URLの中から選択してください。
                """
                
                # データ抽出を実行
                extraction_result = Runner.run_streamed(
                    triage_agent,
                    data_extraction_instruction,
                )
                extraction_data = await handle_stream_events(extraction_result, show_raw_response=False)  # 生のレスポンスを表示しないように設定
                
                if extraction_data is None:
                    raise ValueError("データ抽出に失敗しました")
                    
                return extraction_data
            else:
                # ユーザーが拒否した場合、再入力を要求
                print("カラムの特定に失敗しました。")
                    
                new_input = input("分析したいデータ項目を再入力してください（例: 訪問数,CV率）: ")
                return await QueryAccessData(inputs, wrapper, date_from, date_to, new_input)
        
        # カラムが特定できなかった場合
        print("カラムの特定ができませんでした。")
        if hasattr(columns_json, 'reasoning') and columns_json.reasoning:
            print(columns_json.reasoning)
            
        new_input = input("分析したいデータ項目を再入力してください（例: 日付,訪問数,CV率）: ")
        return await QueryAccessData(inputs, wrapper, date_from, date_to, new_input)

    except json.JSONDecodeError:
        raise ValueError("JSONのパースに失敗しました")
    except Exception as e:
        raise ValueError(f"アクセスデータの抽出中にエラーが発生しました: {str(e)}")
## -----------------------------------------

## レポーティングフェーズ -----------------------
async def GenerateReportingData(inputs: list[TResponseInputItem], wrapper: RunContextWrapper[UserInfo], date_from: str, date_to: str) -> str:
    """レポーティングデータを生成する関数

    Args:
        inputs: 会話履歴のリスト
        wrapper: ユーザー情報を含むコンテキストラッパー
        date_from: レポート対象期間の開始日(YYYY-MM-DD)
        date_to: レポート対象期間の終了日(YYYY-MM-DD)

    Returns:
        str: 生成されたレポート内容

    Raises:
        ValueError: レポート生成に失敗した場合
    """
    try:
        # 入力値のバリデーション
        if not all([wrapper.service_info, wrapper.access_data]):
            raise ValueError("サービス情報またはアクセスデータが不足しています")

        # レポーティング指示を構築
        reporting_instruction = f"""
        # 指示
        以下の情報から、Webサイトのレポーティングをお願いします。

        # コンテキスト
        ## サイトのサービス情報
        {wrapper.service_info}
        ## アクセスデータ
        {wrapper.access_data}
        ## アクセスデータのクエリ期間
        {date_from} から {date_to} です。

        # 期待する出力
        - 一般的なインサイトではなく、サービス情報を元にしたこのサイトならではのインサイトを出力すること。
        """

        # 会話履歴に指示を追加
        inputs.append({"content": reporting_instruction, "role": "user"})

        # レポート生成を実行
        response = Runner.run_streamed(
            dataAnalyst_agent,
            reporting_instruction,
        )
        # レポート部分はストリーミング表示する
        result = await handle_stream_events(response, show_raw_response=True)  # レポート部分は表示する
        print("========================\n")

        if result is None:
            raise ValueError("レポート生成に失敗しました")

        return result

    except Exception as e:
        raise ValueError(f"レポート生成中にエラーが発生しました: {str(e)}")
## -----------------------------------------

async def main():
    # 会話を一意に識別するIDを生成
    conversation_id = str(uuid.uuid4().hex[:16])

    # 実行時の日付を取得
    current_date = datetime.now().strftime("%Y-%m-%d")

    # 会話履歴の初期化 - ユーザーからの最初の入力を追加
    # 現在の日付をコンテキストとして追加
    date_context = f"現在の日付は {current_date} です。"
    inputs: list[TResponseInputItem] = [{"content": date_context, "role": "user"}]

    # ユーザー情報の初期化
    # site_idは仮の値"111"を設定
    # service_info, access_data, report_resultは空文字で初期化
    user_info = UserInfo(
        execution_date=current_date,
        site_id="111",  # TODO: 実際のsite_idを設定する
        service_info="",
        access_data="",
        report_result="",
    )

    try:
        with trace("Marketing Discussion"):
            ## クエリ日付特定フェーズ -----------------------
            # ユーザーからレポート対象期間を取得
            msg = input("> AI: Webサイトのレポーティングを開始します。どの期間のデータをもとにレポーティングしますか？\n> ユーザー: ")
            # ユーザーの入力を会話履歴に追加
            inputs.append({"content": msg, "role": "user"})

            try:
                query_date_result = await identifyQueryDate(inputs, user_info)
                
                # 日付が取得できなかった場合は処理を終了
                if not query_date_result:
                    raise ValueError("> System: 日付の取得に失敗しました")
                
                # 日付のバリデーション
                try:
                    datetime.strptime(query_date_result.date_from, '%Y-%m-%d')
                    datetime.strptime(query_date_result.date_to, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("> System: 不正な日付形式です")
                
                # 開始日が終了日より後の場合はエラー
                if query_date_result.date_from > query_date_result.date_to:
                    raise ValueError("> System: 開始日が終了日より後の日付になっています")
                    
                date_from = query_date_result.date_from
                date_to = query_date_result.date_to

            except ValueError as e:
                print(f"> System: クエリ日付特定時にエラーが発生しました: {str(e)}")
            ## -----------------------------------------
            
            ## サイト情報抽出フェーズ -----------------------
            try:
                query_service_result = await QueryServiceInfo(inputs, user_info)
                # デバッグ出力
                # print(f"サイト情報抽出フェーズの結果: {query_service_result}")
                
                # サイト情報が取得できなかった場合は処理を終了
                if not query_service_result:
                    raise ValueError("> System: サイト情報の取得に失敗しました")
                
                # サイト情報をinputsに追加
                inputs.append({"content": query_service_result, "role": "user"})
                # サイト情報を保存
                user_info.service_info = query_service_result

            except ValueError as e:
                print(f"> System: サイト情報抽出時にエラーが発生しました: {str(e)}")
            ## -----------------------------------------
            
            ## アクセスデータ抽出フェーズ -----------------------
            try:
                # カラム一覧を表示
                print("\n=== 利用可能なカラム ===")
                print("- ページタイトル: Webページのタイトル") 
                print("- URL: 対象ページのURL")
                print("- 訪問数: 対象ページへの訪問回数")
                print("- 直帰率: ページからすぐに離脱したユーザーの割合")
                print("- 平均滞在時間: ページに滞在した平均時間")
                print("- CV数: コンバージョンに至った件数")
                print("- CV率: コンバージョン率")
                print("========================\n")

                # 抽出カラムを入力
                request_text = input("分析したいデータ項目を入力してください（例: 訪問数,CV率）: ")
                
                # 入力値の検証
                if not request_text.strip():
                    raise ValueError("> System: データ項目が入力されていません")

                # アクセスデータを抽出（ユーザー確認機能付き）
                query_access_data_result = await QueryAccessData(inputs, user_info, date_from, date_to, request_text)

                # アクセスデータをコンテキストに追加
                query_access_data_inputs = f"""
                # 分析対象期間
                {date_from} から {date_to}

                # 抽出データ
                {query_access_data_result}
                """

                # コンテキストを更新
                inputs.append({"content": query_access_data_inputs, "role": "user"})
                user_info.access_data = query_access_data_result

            except ValueError as e:
                print(f"\n> System: アクセスデータの抽出に失敗しました")
                print(f"理由: {str(e)}")
                print("入力内容を確認して再度お試しください")
            ## -----------------------------------------

            ## レポーティングフェーズ -----------------------
            try:
                # レポーティングデータを生成
                print("\n=== 生成したレポート ===")
                report_result = await GenerateReportingData(inputs, user_info, date_from, date_to)

                if not report_result:
                    raise ValueError("> System: レポートの生成に失敗しました")

                # レポーティングデータをコンテキストに追加
                reporting_context = f"""
                # 生成されたレポート
                {report_result}
                """
                inputs.append({"content": reporting_context, "role": "user"})

                # レポーティングデータを保存
                user_info.report_result = report_result

                # レポート内容は既にGenerateReportingData関数内で表示されているので、ここでは表示しない
                print("レポートの生成が完了しました")
                print("========================\n")

            except ValueError as e:
                print(f"\n> System: レポート生成に失敗しました")
                print(f"理由: {str(e)}")
                print("入力内容を確認して再度お試しください")
            ## -----------------------------------------

    except Exception as e:
        print(f"\n> System: 予期せぬエラーが発生しました: {str(e)}")    
        print("もう一度お試しください。")
        return


if __name__ == "__main__":
    asyncio.run(main())

# ハイグレードモデルのPCページは、平均滞在時間が他２モデルよりも比較的長いが、CV数は少ない
# ミドルモデルのPCページは最も訪問が少ない
# エントリーモデルのPCページは他２モデルよりも直帰率は低く、CV率が高い
# 営業直通ページは、直帰率が７割
# BtoBサービスページは、平均滞在時間が短く、直帰率が高い