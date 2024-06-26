# 使用Airbox (SG2300X inside) 实现一秒内图像生成（支持文生图、图生图）

## 简介
这个项目提供了一个基于 Gradio 的 Web 界面，允许用户通过文本描述或上传图片来生成图像。用户还可自行替换出图模型（使用本项目中的脚本自行转换Civitai等模型checkpoint为bmodel即可）**注意：本次更新后不会默认加载模型，必须先选择需要的模型进行加载**。

- **文本到图像（Text-to-Image）**：输入描述性文本，系统将根据文本生成相应的图像。
  <img width="681" alt="9000e7dbcee891ba3c90de91b87f7b2" src="https://github.com/ZillaRU/SD-lcm-tpu/assets/25343084/be075231-60a8-4d9e-a21f-25882bcb3177">


- **图像到图像（Image-to-Image）**：上传一张参考图片，系统将结合文本描述生成新的图像。
  <img width="683" alt="b39439e55b9191d76fdf0c37ef9edaa" src="https://github.com/ZillaRU/SD-lcm-tpu/assets/25343084/9e6bf3b4-ee24-43c8-97c8-ee0e391521a6">

- **自定义随机种子（Custom Seed）**：设置一个自定义的随机种子，以获得可复现的生成结果。
- **去噪控制（Denoising Control）**：通过滑块调整去噪强度，改善生成图像的质量。
- **负面提示词 (Negative Prompts)**：通过负面提示器优化图像
- **CFG (Classifier-Free Guidance)**： 通过滑块调整 CFG 值，控制生成结果与输入提示的匹配程度。
- **多比例选择**： 通过选择比例决定生成图像的尺寸。
- **调度器**： 提供多种常见的调度器。
- **模型选择**：用户可自行切换模型。
- **Controlnet**：用户可自行搭配使用 controlnet

## 如何使用

1. **建立模型目录，下载默认模型并安装依赖**：在终端中运行 `bash prepare.sh`。
2. **运行项目**：在终端中运行 `bash run.sh`
3. **访问界面**：在浏览器中打开 `http://ip:8999`（默认端口为 8999，如果端口被占用，Gradio 会自动选择一个可用端口）。
4. **输入文本**：在文本框中输入你的描述；或者同时给出描述和参考图片。
5. **调整参数**：使用滑块调整去噪强度，输入一个自定义的随机种子（可选）。
6. **生成图像**：点击“生成”按钮，系统将根据你的输入生成图像。

## 自行选择出图底模并转换
C站等平台有大量优秀的SD1.5模型，参考`model_export`文件夹中的介绍和脚本，可轻松将任何SD1.5模型转换为bmodel

## 注意事项
- 如果你遇到`GradioUnusedKwargWarning`等问题，请检查你的 Gradio 版本是否为最新。
- 若出现生成图片为黑图，可能是电压问题。
- 测试中部分随机数下出图效果不佳，为偶发问题，排查中。


## changes 
0402: support controlnet and x86 platform

## TODO 
0. 更便捷的一键转换脚本，带有图形化界面 TT ~~4.8~~ 4.11 over (图形化界面目前暂不release)
1. 更方便的lora切换 TT ~~4.12~~ 4月底
2. 更多的controlnet TT ~~4.12~~ 4月底
3. controlnet和sd的权重做融合 未知
4. cfg & no cfg 在各个scheduler上验证 TT 4.3 over  
5. 更多的例子  
6. controlnet : my prompt is more important | controlnet is more important  ~~4.12~~ 4月底

