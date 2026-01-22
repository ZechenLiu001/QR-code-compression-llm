# 内置字体

本目录用于存放内置字体文件，确保跨机器渲染一致性。

## 推荐字体

**DejaVu Sans Mono** (OFL 开源协议)

下载地址: https://dejavu-fonts.github.io/Download.html

将 `DejaVuSansMono.ttf` 放入本目录即可。

## 备用方案

如果未找到字体文件，系统会尝试以下备选：
1. 系统安装的 `DejaVuSansMono.ttf`
2. PIL 默认字体

## 安装命令

```bash
# macOS (Homebrew)
brew install font-dejavu

# Ubuntu/Debian
sudo apt-get install fonts-dejavu-core

# 手动下载
wget https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip
unzip dejavu-fonts-ttf-2.37.zip
cp dejavu-fonts-ttf-2.37/ttf/DejaVuSansMono.ttf assets/fonts/
```
