# 离线镜像

发布包为 LTH_image.tar，标签 localhost/spaceqa-agent:0.6.1-final-r4-20261110。先核验 archive.sha256，再 load。完整说明见 ../02_部署与使用说明.md。Compose 必须设置输入、输出、包、载荷、新运行名及空闲 GPU；没有 build 配置。Dockerfile 的上下文为交付根，仅在已加载冻结镜像基础上更新源码。
