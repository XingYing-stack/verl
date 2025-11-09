#!/usr/bin/env bash
# 一键 SSH 隧道（激进自动重连版）: start | stop | status | restart

# ====== 基本连接信息（按需改）======
HOST="106.63.100.0"
PORT="10029"
USER="gongziqin"
IDENTITY="/workspace/.ssh/docker_rsa"

# 本地端口转发规则（同一条 SSH 连接里复用多条 -L；按需增删）
# 如需对外可见，把 "-L 8888:..." 改为 "-L 0.0.0.0:8888:..."
FORWARDS=(
  "-L 8888:localhost:8888"
  "-L 8000:localhost:8000"
)

# 运行与日志
RUNDIR="./.ssh_tunnel"
PID_FILE="${RUNDIR}/tunnel.pid"
LOG_FILE="${RUNDIR}/tunnel.log"
mkdir -p "${RUNDIR}"

# ====== autossh 监控端口（探测“半断开”）======
MON_PORT=20000   # 如同机起多实例，请为每个实例改成不同值

# ====== 激进的 autossh/ssh 参数（更快判死&重连）======
export AUTOSSH_GATETIME=0
export AUTOSSH_FIRST_POLL=5
export AUTOSSH_POLL=5
export AUTOSSH_LOGFILE="${LOG_FILE}"
export AUTOSSH_LOGLEVEL=7

SSH_OPTS=(
  -p "${PORT}"
  -i "${IDENTITY}"
  -o ServerAliveInterval=10      # 10s 发一次心跳
  -o ServerAliveCountMax=1       # 连续 1 次失败即退出（≈10s 判死）
  -o ConnectTimeout=5            # 建连 5s 超时
  -o ConnectionAttempts=1        # 失败不反复卡住
  -o ExitOnForwardFailure=yes    # 任一 -L 失败立即退出，让 autossh 重试
  -o StrictHostKeyChecking=accept-new
  -N                             # 不执行远程命令，仅转发
)

join_forwards() {
  local arr=()
  for f in "${FORWARDS[@]}"; do arr+=(${f}); done
  echo "${arr[@]}"
}

start() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "已在运行 (PID: $(cat "${PID_FILE}"))"; exit 0
  fi

  echo "启动 SSH 隧道 -> ${USER}@${HOST}:${PORT}"
  echo "日志: ${LOG_FILE}"

  if command -v autossh >/dev/null 2>&1; then
    AUTOSSH_GATETIME=0 AUTOSSH_FIRST_POLL=5 AUTOSSH_POLL=5 \
    AUTOSSH_LOGFILE="${LOG_FILE}" AUTOSSH_LOGLEVEL=7 \
    autossh -M "${MON_PORT}" $(join_forwards) "${SSH_OPTS[@]}" "${USER}@${HOST}" \
      >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    disown || true
  else
    echo "未检测到 autossh，使用 ssh 守护循环作为兜底。" | tee -a "${LOG_FILE}"
    (
      set -e
      while true; do
        echo "[$(date '+%F %T')] 启动 ssh 隧道..." >> "${LOG_FILE}"
        ssh $(join_forwards) "${SSH_OPTS[@]}" "${USER}@${HOST}" >> "${LOG_FILE}" 2>&1 || true
        echo "[$(date '+%F %T')] 连接断开，1 秒后重连..." >> "${LOG_FILE}"
        sleep 1
      done
    ) &
    echo $! > "${PID_FILE}"
  fi

  sleep 1
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "启动成功 (PID: $(cat "${PID_FILE}"))"
  else
    echo "启动失败，查看日志: ${LOG_FILE}"; exit 1
  fi
}


stop() {
  if [ ! -f "${PID_FILE}" ]; then echo "未运行。"; exit 0; fi
  PID=$(cat "${PID_FILE}")
  if kill -0 "${PID}" 2>/dev/null; then
    echo "停止进程 ${PID} ..."
    kill "${PID}" 2>/dev/null || true
    sleep 1
    kill -9 "${PID}" 2>/dev/null || true
  else
    echo "进程 ${PID} 不存在。"
  fi
  rm -f "${PID_FILE}"
  echo "已停止。"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "运行中 (PID: $(cat "${PID_FILE}"))"
  else
    echo "未运行"
    [ -f "${PID_FILE}" ] && echo "(发现失效 PID 文件: ${PID_FILE})"
  fi
  echo "监听端口："
  for f in "${FORWARDS[@]}"; do
    # 从 "-L 8888:localhost:8888" 提取本地端口 8888（或 0.0.0.0:8888）
    local spec="${f#-L }"       # 去 "-L "
    local left="${spec%%:*}"    # 可能是 "8888" 或 "0.0.0.0"
    local lp
    if [[ "${left}" =~ ^[0-9]+$ ]]; then
      lp="${left}"
    else
      # 形如 "0.0.0.0:8888"
      lp="${spec#*:}"; lp="${lp%%:*}"
    fi

    if command -v ss >/dev/null 2>&1; then
      ss -lntp 2>/dev/null | grep -q ":${lp} " && echo "  - ${lp} 已监听" || echo "  - ${lp} 未监听"
    else
      netstat -lnt 2>/dev/null | grep -q ":${lp} " && echo "  - ${lp} 已监听" || echo "  - ${lp} 未监听"
    fi
  done
  echo "日志：${LOG_FILE}"
}

case "$1" in
  start)   start   ;;
  stop)    stop    ;;
  status)  status  ;;
  restart) stop; start ;;
  *) echo "用法: $0 {start|stop|status|restart}"; exit 1 ;;
esac
