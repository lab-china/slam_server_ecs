"""
設定情報を保持するモジュール.
"""

import json
import os

class Settings:
    """Serverで使用する設定情報をまとめたクラス.

    設定ファイルで固定値を一括管理するため,
    config.jsonの内容をsettings.pyで変数化

    """
    with open("./data/config.json", "r") as file:
        data = json.load(file)
        open_mvs_path = data["open_mvs_path"]
        open_mvg_path = data["open_mvg_path"]
        sensor_width_database = data["sensor_width_database"]
        django_server_ip_port = data["django_server_ip_port"]
        strorage_address = data["strorage_address"]
