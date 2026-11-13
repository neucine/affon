# Affon

[English](README.md)

Affon 是一个面向科学计算和机器学习的 TypeScript 运行时。你可以用它描述计算过程（Program），再交给 Session 执行；求导、优化、数据处理和保存训练状态也在同一套工具中。

Affon 仍处于实验阶段。它希望让模型的结构和执行过程更容易看清：一个张量扮演什么角色、各维度代表什么、状态如何变化、计算最终交给哪个后端。项目主要在 CPU 和 Metal 上验证；CUDA 的测试还不够全面，也不以兼容全部 PyTorch API 为目标。方向与验收标准见[路线图](docs/roadmap.md)。

## 快速开始

在 macOS 或 Linux 上安装最新版本：

```bash
curl -fsSL https://affon.ai/install.sh | bash
```

安装后，`affon` 位于 `~/.affon/bin`，安装程序也会将该目录加入 `PATH`。Windows x64 的安装方式见[英文 README](README.md#quick-start)。

进入本仓库后，运行一个不需要准备数据的 CPU 示例：

```bash
affon examples/first-program.ts
```

运行结果会显示输入要求、优化器如何更新状态、逐步下降的损失，以及接近 `[2, 4, 6]` 的预测值。[入门教程](docs/getting-started/first-program.md) 解释了这个例子。

## 目前支持什么

- `affon:core` 包含 Program、Session、求导和变换；`affon:nn`、`affon:ops`、`affon:optim` 分别提供模型组件、运算和优化器。
- 它只支持部分 ESM 模块加载方式，不能直接当作通用的 Node.js 运行时使用。
- Metal 只加速部分运算。CUDA 支持部分使用 `f32` 和 `bf16` 的 Program；实际能否运行，请用目标设备和具体输入形状验证。

平台要求、功能和已知限制详见[英文 README](README.md)。有问题或建议，欢迎到 [GitHub Issues](https://github.com/neucine/affon/issues) 交流。
