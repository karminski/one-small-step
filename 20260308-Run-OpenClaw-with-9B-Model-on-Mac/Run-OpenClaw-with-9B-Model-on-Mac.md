# 9B大模型也能玩转OpenClaw? Mac 部署全教程

简单来讲, Qwen 新推出的 Qwen3.5 9B, 35B-A3B, 27B 模型都支持一定程度的工具调用, 且模型能力对于日常任务来说也够用, 那么作为 OpenClaw 使用的模型也是可以的.

我也验证了一下, 测试在 OpenClaw 接入的情况下, 无论是多模态的图片输入还是工具调用都挺流畅的.

这篇文章准确的来讲是给你的AI看的, 如果你配置卡住了, 让 AI 看这篇文章, 然后帮你配置就可以了, 甚至如果你本地没有 OpenClaw, 你可以本地开一个 claude code, 让 claude code 参考我这篇文章帮你在 Mac 上部署.

## 背景

OpenClaw 是个人 AI 助手框架, 可以接各种大模型 API. 我想让它接我 Mac 上用 mlx_vlm 跑的本地模型 (Qwen3.5 系列), 这样不用花钱, 也不用担心隐私.

推理框架用的是 mlx_vlm, 它提供了一个 OpenAI 兼容的 HTTP API. 理论上配个 baseUrl 就能用. 

## 第零步: 下载模型

直接去huggingface 下载模型, 比如 https://huggingface.co/mlx-community/Qwen3.5-9B-8bit, 如果你的 Mac 内存比较小也可以考虑 4bit 版本. 如果你是 windows 或者 linux, 可以用 GGUF 搭配 llama.cpp.

## 第一步: 创建共享 venv

多个 Qwen3.5 模型可以共用一个 venv, 不用把 20G 的依赖复制三份. venv 里装的是推理框架, 模型权重是独立的文件, 通过 `--model` 参数指定就行.

```bash
# 我的模型放在了 /Volumes/WORK_2/models, 可以根据你自己的的实际情况调整
python3 -m venv /Volumes/WORK_2/models/Qwen3.5-venv
source /Volumes/WORK_2/models/Qwen3.5-venv/bin/activate
pip install git+https://github.com/Blaizzy/mlx-vlm.git@main # 这里一定要用 github 上的最新版本, Pypi 上的版本还不支持 tool call.
pip install torch torchvision # 这两个都要装, 是 mlx-vlm 必须的库
```

**踩坑: venv 不能复制!** Python venv 里的脚本 (pip, python3 等) 都硬编码了原始路径的 shebang. 你把 `.venv` 文件夹复制到别的地方, 里面的 pip 还是指向旧路径, 装的包全装到旧位置去了. 表面上 `(.venv)` 提示符亮着, 实际上 `which pip` 指向的是系统 Python. 解决办法: 永远在目标位置重新 `python3 -m venv` 创建, 别复制.

## 第二步: 用 pm2 管理模型服务

先检查是否安装了 node.js 环境, 如果没有先安装 node.js.

关键技巧: 不需要 `source activate`, 直接用 venv 里的 Python 完整路径:

```bash
pm2 start /Volumes/WORK_2/models/Qwen3.5-venv/bin/python3 \
  --name "qwen3.5-api" \
  --interpreter none \
  -- -m mlx_vlm.server \
  --model /Volumes/WORK_2/models/Qwen3.5-9B-8bit \
  --host 0.0.0.0 --port 10012 --trust-remote-code
```

`source activate` 本质就是把 venv 的 bin 目录加到 PATH 前面, 直接用完整路径调 Python 效果完全一样.

## 第三步: 配置 OpenClaw

在 OpenClaw 的 `config.json` 里添加 local provider:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "local": {
        "baseUrl": "http://10.0.6.26:10012",
        "apiKey": "DEADBEEF",
        "api": "openai-completions",
        "models": [
          {
            "id": "/Volumes/WORK_2/models/Qwen3.5-35B-A3B-8bit",
            "name": "Qwen 3.5 35B A3B",
            "reasoning": false,
            "input": ["text", "image"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 262144,
            "maxTokens": 32768
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "models": {
        "local//Volumes/WORK_2/models/Qwen3.5-35B-A3B-8bit": {
          "alias": "qwen3.5-35b",
          "streaming": false
        }
      },
      "timeoutSeconds": 600
    }
  }
}
```

几个注意点:
- `apiKey` 设成 `"DEADBEEF"` 或其他的什么随机字符串, 本地模型不需要鉴权, 以防 OpenClaw 不兼容留空.
- model id 就是模型权重的完整路径, 因为 mlx_vlm 的 API 用路径来区分模型.
- `"streaming": false` 关掉流式传输, 避免 SSE 兼容性问题导致超时. (打开应该也没事)
- `"timeoutSeconds": 600` 给本地模型 10 分钟响应时间.
- 模型引用格式是 `local//Volumes/...`, 第一个 `/` 是 provider 和 model id 的分隔符, 第二个 `/` 是路径本身.

## 第四步: 可能的 Discord 权限配置问题

**如果你运行: `/model` 命令返回 "You are not authorized to use this command"**

这不是 Discord Bot 权限问题, 是 OpenClaw 的鉴权. 即使 `groupPolicy` 设成 `"open"`, slash 命令仍然需要在 guild 配置里明确授权你的 User ID:

```json
{
  "channels": {
    "discord": {
      "groupPolicy": "open",
      "guilds": {
        "你的SERVER_ID": {
          "requireMention": false,
          "users": ["你的USER_ID"]
        }
      }
    }
  }
}
```

User ID 和 Server ID 在 Discord 开启开发者模式后右键复制.

## 调试技巧

如果报错 HTTP 422 大概率是你用的 mlx-vlm 不够新, 请使用 github 版本而不是 pip 安装的旧版, 如果不知道是什么问题可以按照下面的思路 debug:

**用 socat 抓 OpenClaw 发出的原始请求:**
```bash
pm2 stop qwen3.5-api  # 先释放端口
socat -v TCP-LISTEN:10012,reuseaddr,fork \
  SYSTEM:'cat; echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"'
```

**用 curl 逐个字段排查哪个触发 422:**
```bash
# 基础测试
curl -X POST http://10.0.6.26:10012/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"stream":false}'

# 加上 tools 测试
curl -X POST ... -d '{"model":"...","messages":[...],"stream":false,"tools":[]}'

# 加上 content 数组格式测试
curl -X POST ... -d '{"model":"...","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stream":false}'
```

## 总结

总之不要自己动手解决问题, 尽量让AI帮你搞, 省时间. 
