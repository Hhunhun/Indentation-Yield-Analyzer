import os

# 核心设置：在加载 pysr/julia 之前，将 Julia 的包服务器强制指向北京外国语大学镜像（或清华镜像）
os.environ["JULIA_PKG_SERVER"] = "https://mirrors.bfsu.edu.cn/julia"
# 备用清华源：os.environ["JULIA_PKG_SERVER"] = "https://mirrors.tuna.tsinghua.edu.cn/julia"

import pysr

# 触发依赖包的编译与安装
pysr.install()