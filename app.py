# 项目一键启动脚本
# 作用：同时拉起后端（FastAPI/uvicorn，8000 端口）和前端（Vite/React，5173 端口），
#       并负责两个进程的协调：端口占用检测（已跑的复用）、日志加前缀区分、
#       退出时互相清理（按 Ctrl+C 或任一端挂掉，另一个一起关，不留孤儿进程）。
# 运行：python app.py （Windows/Linux/Mac 通用）

import os
import socket
import subprocess
import sys
import threading
import time

# 项目根目录（app.py 所在目录）：uvicorn 必须从这里跑，才能 import 到 main/core/agent
ROOT = os.path.dirname(os.path.abspath(__file__))
# 前端目录：vite 必须从这里跑
FRONTEND_DIR = os.path.join(ROOT, "frontend", "web")

BACKEND_PORT = 8000   # 后端 uvicorn 端口
FRONTEND_PORT = 5173  # 前端 vite 端口


def _port_in_use(port: int) -> bool:
    """检查端口是否已被占用：能连上说明已经有服务在跑"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _npm_cmd() -> list[str]:
    """返回当前系统启动 vite 的命令：Windows 用 shell 跑 npm.cmd，其他平台直接 npm"""
    if os.name == "nt":
        return ["npm.cmd", "run", "dev"]
    return ["npm", "run", "dev"]


def _pipe_reader(stream, name: str) -> None:
    """读子进程输出并加前缀打印到终端，两个服务的日志不会混成一团分不清"""
    for line in stream:
        print(f"[{name}] {line}", end="", flush=True)


def _spawn(cmd: list[str], cwd: str, name: str):
    """启动一个子进程：输出重定向到内存管道，由 _pipe_reader 加前缀转发出来"""
    p = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    threading.Thread(target=_pipe_reader, args=(p.stdout, name), daemon=True).start()
    return p


def _kill_proc(proc, name: str) -> None:
    """结束子进程：Windows 用 taskkill 连进程树一起杀（npm 会派生子进程），其他平台直接 terminate"""
    if proc.poll() is not None:  # 已经退出了就不用管
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    print(f"[{name}] 已停止")


def main():
    """主入口：检测端口 → 启动没跑的服务 → 监控进程，任一退出就清理另一个"""
    # 1. 先看两个端口，已经有人占用的就复用，不重复启动
    need_back = not _port_in_use(BACKEND_PORT)
    need_front = not _port_in_use(FRONTEND_PORT)

    procs = []  # 记录本脚本启动的子进程，退出时要清理
    if not need_back and not need_front:
        print("前后端都已在运行，直接用浏览器打开 http://localhost:5173/")
        return

    # 2. 启动后端 uvicorn（用当前 python 解释器，保证 uvicorn 在同一个环境里）
    if need_back:
        print(f"启动后端（{BACKEND_PORT} 端口）…")
        procs.append(("后端", _spawn(
            [sys.executable, "-m", "uvicorn", "frontend.server:app", "--port", str(BACKEND_PORT)],
            ROOT, "后端")))
    else:
        print(f"后端已占用 {BACKEND_PORT} 端口，直接复用，不重复启动")

    # 3. 启动前端 vite
    if need_front:
        print(f"启动前端（{FRONTEND_PORT} 端口）…")
        procs.append(("前端", _spawn(_npm_cmd(), FRONTEND_DIR, "前端")))
    else:
        print(f"前端已占用 {FRONTEND_PORT} 端口，直接复用，不重复启动")

    print("=" * 50)
    print("服务地址：")
    print(f"  前端： http://localhost:{FRONTEND_PORT}/")
    print(f"  后端： http://localhost:{BACKEND_PORT}/docs")
    print("按 Ctrl+C 可同时停止本脚本拉起的服务")
    print("=" * 50)

    # 4. 监控：本脚本拉起的任一进程退出，就把另一个也关掉，然后整体退出
    try:
        while any(p.poll() is None for _, p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在清理…")
    finally:
        for name, p in procs:
            _kill_proc(p, name)


if __name__ == "__main__":
    main()
