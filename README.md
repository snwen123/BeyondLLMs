# BeyondLLMs
Some note about LLMs

## LLMs model

Alpaca：https://github.com/tatsu-lab/stanford_alpaca

Vicuna：https://vicuna.lmsys.org/
 - Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality
 - UC Berkeley, CMU, Stanford, and UC San Diego
 
Dolly: https://github.com/databrickslabs/dolly
 - Databricks

Alpaca-CoT：https://github.com/PhoebusSi/Alpaca-CoT/blob/main/CN_README.md
 - 指令数据集比较多

Chenise-llama-alpaca-lora：github.com/zhangnn520/chinese_llama_alpaca_lora
 - 中文LLaMA，信息抽取

PiXiu-貔貅: https://github.com/catqaq/ChatPiXiu

DeepSpeed-Chat：https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
 - 微软，RLHF全流程

Open Assistant：
 - 开源聊天模型Open Assistant正式发布，包含大量AI生成的、人工标注的语料库和包括基于LLaMA和基于Pythia的多种模型可选。发布的数据集包括超过161K较高质量的，多达35种语言的人工助手型交互对话语料库

LMFlow：https://github.com/OptimalScale/LMFlow
 - 港科大，task tuning, instruction tuning, alignment tuning, evaluation, inference，都有，流程比较全面

流萤（中文对话式大模型）：https://github.com/yangjianxin1/Firefly

BELLE：https://github.com/LianjiaTech/BELLE
 - Be Everyone's Large Language model Engine（开源中文对话大模型）

MOSS：github.com/OpenLMLab/MOSS
 - 复旦大学

医疗领域ChatGPT：
 - https://github.com/cambridgeltl/visual-med-alpaca ， 基于LLaMA-7B，含有LoRa版本
 - github.com/kbressem/medAlpaca 基于LLaMA-7B
 - https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese ， 华佗，基于LLaMA的医疗领域微调
 - github.com/SCIR-HI/Med-ChatGLM 基于ChatGML的医疗领域微调

金融领域ChatGPT:
 - https://finchat.io/

TRL和TRL-RLHF-LLaMa：https://huggingface.co/blog/stackllama

## LLMs data
 - 指令数据集合: https://github.com/FreedomIntelligence/InstructionZoo
 - 指令数据生产：https://github.com/togethercomputer/RedPajama-Data
 - 指令数据集：github.com/raunak-agarwal/instruction-datasets

## LLMs finetuning
 - 轻量化微调工具：https://github.com/OpenLMLab/TuneLite  支持Coloss
 - 混合精度训练、DDP、gradient checkpoing：https://zhuanlan.zhihu.com/p/448395808
 - 基于ChatGLM的几种精调方法：https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/README_zh.md

## LLMs evaluation
 - https://github.com/ninehills/llm-playground  可同时比较多个prompt和多个chatGPT的输出

## LLMs pruners:
LLMs裁剪工具：https://github.com/yangjianxin1/LLMPruner

## LLMs usage:
 - 使用自然语言绘制流程图：https://github.com/fraserxu/diagram-gpt
 - github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide  大模型构建应用
 - github.com/yanqiangmiffy/Chinese-LangChain  大模型构建应用
 - https://github.com/plchld/InsightFlow  LLM用于转录、翻译、分析网页等应用。基于AI技术的解决方案，可从视频、文档等中提取有价值的洞察，实现即时、上下文感知的聊天式查询

## LLMs interpretability

 Interpretable Unified Language Checking
    动机：大型语言模型(Large Language Models, LLM)存在非事实、有偏见和仇恨言论等不良行为，需要开发一种可解释、统一的语言检查方法。方法：提出一种基于简单、少样本、统一提示的“1/2-shot”多任务语言检查方法，该方法能同时处理事实核查、刻板印象检测和仇恨言论检测任务，将语言伦理建模与事实检查相结合，利用前向推理技术来提高结果准确性。优势：所提出的UniLC方法在多个语言任务上的表现优于全监督的基线模型，表明基于强大的潜知识表示，LLM可以成为检测虚假信息、刻板印象和仇恨言论的自适应和可解释工具。
    
 What does ChatGPT return about human values? Exploring value bias in ChatGPT using a descriptive value theory
   用描述性价值理论探索ChatGPT的价值偏差
 
## LLMs with security, privacy 
 - 《Toxicity in ChatGPT: Analyzing Persona-assigned Language Models》A Deshpande, V Murahari, T Rajpurohit, A Kalyan, K Narasimhan [Princeton University & The Allen Institute for AI & Georgia Tech] (2023)
 - 《Multi-step Jailbreaking Privacy Attacks on ChatGPT》H Li, D Guo, W Fan, M Xu, Y Song [Hong Kong University of Science and Technology & Peking University] (2023)
   
## 资料汇总
1. 类ChatGPT模型汇总：https://github.com/chenking2020/FindTheChatGPTer
2. 类ChatGPT模型汇总（模型、数据、评估）：https://github.com/FreedomIntelligence/LLMZoo
3. 中文图像数据标注工具：https://github.com/opendatalab/labelU/blob/release/v0.5.5/README_zh-CN.md
4. 类ChatGPT模型，ChatGPT相关论文汇总：https://github.com/MLNLP-World/Awesome-LLM
5. https://blog.replit.com/llm-training: How to train your own Large Language Models
