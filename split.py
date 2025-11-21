from __future__ import annotations

from typing import Iterable, Sequence, Union, Literal
import hashlib

import pandas as pd


def _stable_hash_to_group(key: str, n_groups: int = 100) -> int:
    """
    key（文字列）を安定的な整数 [0, n_groups-1] に写像するヘルパー関数。
    Python 組み込みの hash() は実行ごとに変わるので使わない。
    """
    md5 = hashlib.md5(key.encode("utf-8")).hexdigest()
    as_int = int(md5, 16)
    return as_int % n_groups

def create_sample_column(
    df: pd.DataFrame,
    id_cols: Union[str, Sequence[str]],
    train_fraction: float = 0.8,
    method: Literal["hash", "mod"] = "hash",
    n_groups: int = 100,
) -> pd.DataFrame:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1 (exclusive).")

    # id_cols をリストに正規化
    if isinstance(id_cols, str):
        id_cols_list = [id_cols]
    else:
        id_cols_list = list(id_cols)

    # 列が存在するかチェック
    missing = set(id_cols_list) - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    threshold = int(train_fraction * n_groups)

    df_out = df.copy()

    if method == "mod":
        if len(id_cols_list) != 1:
            raise ValueError(
                "method='mod' を使う場合は id_cols は 1 列のみを指定してください。"
            )
        col = id_cols_list[0]

        # 念のため整数化（必要に応じて調整）
        # 数値でない場合はエラーになるので、事前に型を揃えておくのが安心です
        ids = df_out[col].astype("int64")
        groups = (ids % n_groups).to_numpy()

    elif method == "hash":
        # 複数列もOK: 文字列として結合してからハッシュ
        key_series = (
            df_out[id_cols_list]
            .astype(str)
            .agg("||".join, axis=1)  # 行ごとに "col1||col2||..." というキーを作る
        )
        groups = key_series.map(lambda x: _stable_hash_to_group(x, n_groups)).to_numpy()

    else:
        raise ValueError("method must be either 'hash' or 'mod'.")

    # グループ番号 [0, n_groups-1] を 'train' / 'test' に割り当て
    is_train = groups < threshold
    df_out["sample"] = "test"
    df_out.loc[is_train, "sample"] = "train"

    return df_out