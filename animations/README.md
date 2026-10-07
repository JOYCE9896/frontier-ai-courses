# 动画源代码

网站里的动画用 [Manim Community](https://www.manim.community/) 制作，Manim 最初是 3Blue1Brown 为自己的视频开发的动画引擎。

| 文件 | 动画 | 用在 |
| :--- | :--- | :--- |
| `attention.py` | 注意力的四个步骤 | AI 基础第 12 篇 |
| `kvcache.py` | 有无 KV 缓存时每步的计算量 | CMU 11-664 第 20 讲补充篇 |

## 重新生成

```bash
brew install cairo pango pkg-config   # macOS
pip install manim
manim -qm --format mp4 attention.py Attention
```

文字使用 macOS 自带的宋体（Songti SC），在其他系统上需要把脚本开头的 `FONT` 换成本机已有的中文字体。
