# Huggingface-practice

- 图像分类：利用ViT模型，将classifier层的输出维度改为100并在cifar-100数据集上进行linear-probe微调，最终达到0.83正确率
- 图文匹配：利用CLIP模型，将vit-model部分的参数冻结，利用COCO数据集部分图片匹配的中文句子做训练集，最后用中文句子测试图片，微调后模型对两句中文的区分度从0.3变为0.94
- 看图说话，图片问答：分别利用BLIP-2和BLIP完成上述任务
