# 地图资源迁移

仓库包含 [Lotus 地图包](../assets/lotus-assets.zip) 和 [SHA-256 文件](../assets/lotus-assets.sha256)，普通 `git clone` / `git pull` 会下载，无需 Git LFS。环境需要解包后的 `maps/lotus/manifest.json`、`collision.json` 和 manifest 列出的 GLB 文件；缺少任意一项都会停止启动。数据来自已安装 CA-VAL-AIM 的可选地图资源，归属见 [资源说明](../assets/NOTICE.md)。

## 安装仓库自带资源包

`assets/lotus-assets.zip`：73,856,830 字节。

SHA-256：

```text
309ae815598e3180e626eac0fbb693555abce65740e877fb3e40e0530d3746a2
```

在仓库根目录执行以下命令，无需再从工作电脑复制文件：

```bash
python scripts/assets.py install --archive assets/lotus-assets.zip \
  --sha256 309ae815598e3180e626eac0fbb693555abce65740e877fb3e40e0530d3746a2
python scripts/bootstrap.py --maps-dir assets/maps-data --skip-build
```

安装会拒绝覆盖非空目标目录，并逐文件校验包内清单。`--maps-dir` 指向 **包含 maps 子目录的目录**，不是 `maps/lotus`。

## 从自己的训练器打包

以下 `TRAINER_ROOT` 是训练器安装目录，里面应有 `maps/lotus`：

```bash
python scripts/assets.py pack --root TRAINER_ROOT --map lotus \
  --output assets/lotus-assets-local.zip
```

命令生成本机 ZIP 和 `.sha256` 文件，只包含指定地图，不打包个人设置或运行记录。ZIP 时间戳可能随源文件改变，以本次命令打印的哈希为准。仓库中的固定版本包保持原样；若输出文件已存在，指定新的 `--output`。

也可以直接使用原安装目录，不复制地图：

```bash
python scripts/bootstrap.py --maps-dir TRAINER_ROOT --skip-build
```

## 从自己的存储拉取

如果有可访问的 HTTPS 资源地址，先把合法来源的包放到自己的存储，再运行：

```bash
python scripts/assets.py fetch --url "$MAP_BUNDLE_URL" \
  --sha256 "$MAP_BUNDLE_SHA256" --destination assets/maps-data
python scripts/bootstrap.py --maps-dir assets/maps-data --skip-build
```

带令牌的地址只通过本机环境变量传入，不写进仓库。工具只下载与校验，不上传资源。

其他地图可以用相同格式打包，但还需要独立验证起点、终点和碰撞。当前默认场景及验收仅覆盖 Lotus。资源或上游地图更换后，重新运行 smoke，不直接沿用训练结果。
