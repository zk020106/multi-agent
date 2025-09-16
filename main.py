"""
AG-UI 服务主入口（根目录）

用法：
  python main.py --host 0.0.0.0 --port 8000 --reload

跨域配置（环境变量）：
  AGUI_CORS_ORIGINS (逗号分隔)，AGUI_CORS_HEADERS，AGUI_CORS_METHODS, AGUI_CORS_CREDENTIALS
"""

import argparse
import os
from typing import List, Any

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from server.ag_ui_adapter import app
from utils.config import load_config


def _split_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(',') if v.strip()]


def configure_cors_from_config(config_path: str) -> None:
    """从 config.yaml 的 custom 字段读取 CORS 配置，不走环境变量。

    支持的 custom 键（可选）：
      - cors_origins: ["http://localhost:5173", "http://127.0.0.1:3000"] 或逗号分隔字符串
      - cors_methods: ["*"] 或逗号分隔
      - cors_headers: ["*"] 或逗号分隔
      - cors_credentials: true/false
    若未配置，则默认允许所有来源与方法/头。
    """
    origins: List[str] = ['*']
    methods: List[str] = ['*']
    headers: List[str] = ['*']
    credentials: bool = False

    try:
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            custom: Any = getattr(cfg, 'custom', {}) or {}
            if isinstance(custom, dict):
                if 'cors_origins' in custom:
                    v = custom['cors_origins']
                    origins = v if isinstance(v, list) else _split_csv(str(v))
                if 'cors_methods' in custom:
                    v = custom['cors_methods']
                    methods = v if isinstance(v, list) else _split_csv(str(v))
                if 'cors_headers' in custom:
                    v = custom['cors_headers']
                    headers = v if isinstance(v, list) else _split_csv(str(v))
                if 'cors_credentials' in custom:
                    credentials = bool(custom['cors_credentials'])
    except Exception:
        # 配置读取失败则使用默认
        pass

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=methods,
        allow_headers=headers,
    )


def main():
    parser = argparse.ArgumentParser(description='AG-UI 服务启动')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', default=False)
    parser.add_argument('--config', default=os.path.join(os.getcwd(), 'config.yaml'))
    args = parser.parse_args()

    configure_cors_from_config(args.config)
    uvicorn.run('server.ag_ui_adapter:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()


