# BeyondLLMs
Some note about LLMs

## general LLMs

 - Alpaca：https://github.com/tatsu-lab/stanford_alpaca
 - OpenAlpaca：github.com/yxuansu/OpenAlpaca 基于OpenLLaMa
 - Vicuna：https://vicuna.lmsys.org/
   
   Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. UC Berkeley, CMU, Stanford, and UC San Diego
 - Dolly: https://github.com/databrickslabs/dolly  Databricks
 - Alpaca-CoT：https://github.com/PhoebusSi/Alpaca-CoT/blob/main/CN_README.md  指令数据集比较多
 - Chenise-llama-alpaca-lora：github.com/zhangnn520/chinese_llama_alpaca_lora  中文LLaMA，信息抽取
 - PiXiu-貔貅: https://github.com/catqaq/ChatPiXiu
 - DeepSpeed-Chat：https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat  微软，RLHF全流程
 - Open Assistant：开源聊天模型Open Assistant正式发布，包含大量AI生成的、人工标注的语料库和包括基于LLaMA和基于Pythia的多种模型可选。发布的数据集包括超过161K较高质量的，多达35种语言的人工助手型交互对话语料库
 - LMFlow：https://github.com/OptimalScale/LMFlow  港科大，task tuning, instruction tuning, alignment tuning, evaluation, inference，都有，流程比较全面
 - 流萤（中文对话式大模型）：https://github.com/yangjianxin1/Firefly
 - BELLE：https://github.com/LianjiaTech/BELLE  Be Everyone's Large Language model Engine（开源中文对话大模型）
 - MOSS：github.com/OpenLMLab/MOSS  复旦大学
 - Panda: github.com/dandelionsllm/pandallm  海外中文开源大语言模型，基于 Llama-7B, -13B, -33B, -65B 进行中文领域上的持续预训练，使用了接近15M条数据，并针对推理能力在中文benchmark上进行了评测
 - Lamini:  https://lamini.ai/   基于统一接口的大语言模型
 - Lit-LLaMA:　https://github.com/Lightning-AI/lit-llama/tree/main
 - CaMA: A Chinese-English Bilingual LLaMA Model - CaMA: A Chinese-English Bilingual LLaMA Model.' ZJUNLP GitHub: github.com/zjunlp/CaMA 。CaMA: 一种支持中英语言的LLaMA模型，通过全量预训练和指令微调提高了中文理解能力、知识储备和指令理解能力
 - CPM-Bee: github.com/OpenBMB/CPM-Bee  'CPM-Bee - 完全开源、允许商用的百亿参数中英文基座模型，采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型
   
 ## LLM微调
 - 来自Lit-LLaMA的lit-stablelm：https://github.com/Lightning-AI/lit-stablelm
 - LLMTune: https://github.com/kuleshov-group/llmtune
   在消费级GPU上微调大型65B+LLM。可以在普通消费级GPU上进行4位微调，例如最大的65B LLAMA模型。LLMTune还实现了LoRA算法和GPTQ算法来压缩和量化LLM，并通过数据并行处理大型模型。此外，LLMTune提供了命令行界面和Python库的使用方式
 - bactrain-x: https://github.com/mbzuai-nlp/bactrian-x  多语言指令遵循模型
 - miniGPT-4:  https://github.com/Vision-CAIR/MiniGPT-4
 - PaLM开源复现 https://github.com/conceptofmind/PaLM
 - 中文增量预训练的LLaMA
   https://github.com/OpenLMLab/OpenChineseLLaMA ， LLaMA-7B
   https://github.com/ymcui/Chinese-LLaMA-Alpaca  ， LLaMA-7B、LLaMA-13B
 - 'Axolotl - Go ahead and axolotl questions' OpenAccess-AI-Collective GitHub: github.com/OpenAccess-AI-Collective/axolotl  一个用于微调的代码库，支持使用不同的模型和注意力机制进行微调
 - 多卡微调ChatGLM-6B， 简单高效实现多卡微调大模型，目前已实现LoRA, Ptuning-v2, Freeze三种微调方式  github.com/CSHaitao/ChatGLM_mutli_gpu_tuning
 - LLaMA全流程微调：PT+STF+RLHF github.com/hiyouga/LLaMA-Efficient-Tuning  

## domain LLMs
医疗领域ChatGPT：
 - https://github.com/cambridgeltl/visual-med-alpaca ， 基于LLaMA-7B，含有LoRa版本
 - github.com/kbressem/medAlpaca 基于LLaMA-7B
 - https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese ， 华佗，基于LLaMA的医疗领域微调
 - github.com/SCIR-HI/Med-ChatGLM 基于ChatGML的医疗领域微调
 - https://github.com/MediaBrain-SJTU/MedicalGPT-zh ，基于ChatGLM-6B的中文医疗领域LoRa微调。中文医疗通用语言模型，基于28个科室的医疗共识与临床指南文本，提高模型的医疗领域知识与对话能力
 - github.com/CogStack/OpenGPT  OpenGPT：用于创建基于指令的数据集并训练对话领域专家大型语言模型(LLMs)的框架。已经成功应用于训练健康护理对话模型NHS-LLM，利用来自英国国家卫生服务体系(NHS)网站的数据，生成了大量的问答对和独特对话。此外，OpenGPT还提供了如何创建并训练一个小型健康护理对话LLM的教程
 - QiZhenGPT: An Open Source Chinese Medical Large Language Model' CMKRG GitHub: github.com/CMKRG/QiZhenGPT  启真医学大模型，利用启真医学知识库构建的中文医学指令数据集，并基于此在LLaMA-7B模型上进行指令精调，大幅提高了模型在中文医疗场景下效果，首先针对药品知识问答发布了评测数据集，后续计划优化疾病、手术、检验等方面的问答效果，并针对医患问答、病历自动生成等应用展开拓展
 - XrayGLM - 首个会看胸部X光片的中文多模态医学大模型 | The first Chinese Medical Multimodal Model that Chest Radiographs Summarization.' WangRongsheng GitHub: github.com/WangRongsheng/XrayGLM

金融领域ChatGPT:
 - https://finchat.io/

法律领域ChatGPT:
 - github.com/LiuHC0428/LAW-GPT 中文法律对话语言模型
 - github.com/CSHaitao/LexiLaw LexiLaw - 中文法律大模型

化学领域ChatGPT:
 - github.com/OpenBioML/chemnlp

数学领域ChatGPT:
 - Goat: 擅长算术任务的LLaMA微调模型  github.com/liutiedong/goat  基于LLaMA

## LLMs和RL
 - TRL和TRL-RLHF-LLaMa：https://huggingface.co/blog/stackllama
 - github.com/floodsung/LLM-with-RL-papers
 - https://github.com/salesforce/HIVE   HIVE: Harnessing Human Feedback for Instructional Visual Editing.
 - RLHF数据集：Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation，面向文本到图像生成的用户偏好开放数据集
 - LLM人工偏好数据集相关资源列表,包括自然语言处理、计算机视觉和音乐生成等. github.com/glgh/awesome-llm-human-preference-datasets
 - github.com/PKU-Alignment/safe-rlhf  安全强化学习人工反馈框架Beaver，保障AI对人类价值观的精准对齐。首个结合安全偏好的RLHF框架，支持多模型训练，数据可复现
 - [CL] Aligning Large Language Models through Synthetic Feedback. 人工合成反馈，据称与RLHF效果相同
 - AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback - A Simulation Framework for RLHF and alternatives' Tatsu's shared repositories GitHub: github.com/tatsu-lab/alpaca_farm  AlpacaFarm: 一个模拟框架，用于人工反馈学习方法的研究和开发，促进指令遵循和对齐的可访问性研究。该框架包括模拟语言模型的偏好反馈、自动评估方法以及基准算法的参考实现
 - [CL] Training Socially Aligned Language Models in Simulated Human Society 在模拟人类社会中训练社会化对齐的语言模型

## milti-model LLMs
 - github.com/open-mmlab/Multimodal-GPT  基于OpenFlamingo多模态模型，使用各种开放数据集创建各种视觉指导数据，联合训练视觉和语言指导，有效提高模型性能
 - 《mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality}》(2023) GitHub: github.com/X-PLUG/mPLUG-Owl
 - LLaVA：https://llava-vl.github.io/  使用GPT-4生成多模态指令
 - https://github.com/DAMO-NLP-SG/Video-LLaMA   视频指令

## LLMs data
 - 指令数据集合: https://github.com/FreedomIntelligence/InstructionZoo
 - 指令数据生产：https://github.com/togethercomputer/RedPajama-Data
 - 指令数据集：github.com/raunak-agarwal/instruction-datasets
 - 指令数据集：https://github.com/voidful/awesome-chatgpt-dataset
 - Medical QA data  https://github.com/abachaa

## LLMs finetuning
 - 轻量化微调工具：https://github.com/OpenLMLab/TuneLite  支持Coloss
 - 混合精度训练、DDP、gradient checkpoing：https://zhuanlan.zhihu.com/p/448395808
 - 基于ChatGLM的几种精调方法：https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/README_zh.md
 - 大语言模型精调技术笔记： https://github.com/ninehills/ninehills.github.io/issues/92
 - LoRa检查器：https://github.com/rockerBOO/lora-inspector 
 - QLoRa在单个GPU上微调65B LLaMA语言模型： 《QLoRA: Efficient Finetuning of Quantized LLMs》T Dettmers, A Pagnoni, A Holtzman, L Zettlemoyer [University of Washington] (2023)

## LLMs for code generation 
 - 基于LLaMA－7B: https://github.com/FSoft-AI4Code/CodeCapybara#humaneval
 - github.com/zxx000728/CodeGPT  CodeGPT: 提高编程能力的关键在于数据。CodeGPT是通过GPT生成的用于GPT的代码对话数据集。现在公开了32K条中文数据，让模型更擅长编程
 - Open Code Large Language Models for Code Understanding and Generation  CodeT5+：面向代码理解和生成的开放大型代码语言模型

## LLMs evaluation
 - https://github.com/ninehills/llm-playground  可同时比较多个prompt和多个chatGPT的输出
 - https://github.com/mbzuai-nlp/LaMini-LM  使用多个nlp下游任务
 - https://github.com/WeOpenML/PandaLM/blob/main/README.md  可复现的自动化评估大语言模型

## LLMs pruners:
LLMs裁剪工具：https://github.com/yangjianxin1/LLMPruner

## LLMs usage:
 - 使用自然语言绘制流程图：https://github.com/fraserxu/diagram-gpt
 - github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide  大模型构建应用
 - github.com/yanqiangmiffy/Chinese-LangChain  大模型构建应用
 - https://github.com/plchld/InsightFlow  LLM用于转录、翻译、分析网页等应用。基于AI技术的解决方案，可从视频、文档等中提取有价值的洞察，实现即时、上下文感知的聊天式查询
 - The Ultimate GPT-4 Guide: ChatGPT和GPT-4用法，用于教学领域等
 - 从接口调用chatGPT，从数据源（pdf, ppt, word, 网页等）中提取答案：https://github.com/geeks-of-data/knowledge-gpt
 - https://github.com/xx025/carrot ChatGPT免费使用站点，使用方法，集成应用
 - https://github.com/emmethalm/infiniteGPT 批量向openai api提示的用法
 - github.com/csunny/DB-GPT DB-GPT：基于vicuna-13b和FastChat的开源实验项目，采用了langchain和llama-index技术进行上下文学习和问答。项目完全本地化部署，保证数据的隐私安全，能直接连接到私有数据库处理私有数据。其功能包括SQL生成、SQL诊断、数据库知识问答等
 - LLMs langchain  https://github.com/ColinEberhardt/langchain-mini
 - DB-GPT: github.com/csunny/DB-GPT  基于vicuna-13b和FastChat的开源实验项目，采用了langchain和llama-index技术进行上下文学习和问答。项目完全本地化部署，保证数据的隐私安全，能直接连接到私有数据库处理私有数据。其功能包括SQL生成、SQL诊断、数据库知识问答等


## LLMs interpretability

 Interpretable Unified Language Checking
    动机：大型语言模型(Large Language Models, LLM)存在非事实、有偏见和仇恨言论等不良行为，需要开发一种可解释、统一的语言检查方法。方法：提出一种基于简单、少样本、统一提示的“1/2-shot”多任务语言检查方法，该方法能同时处理事实核查、刻板印象检测和仇恨言论检测任务，将语言伦理建模与事实检查相结合，利用前向推理技术来提高结果准确性。优势：所提出的UniLC方法在多个语言任务上的表现优于全监督的基线模型，表明基于强大的潜知识表示，LLM可以成为检测虚假信息、刻板印象和仇恨言论的自适应和可解释工具。
    
 What does ChatGPT return about human values? Exploring value bias in ChatGPT using a descriptive value theory
   用描述性价值理论探索ChatGPT的价值偏差
 
 - "According to ..." Prompting Language Models Improves Quoting from Pre-Training Data. 用"According to ..."提示语言模型改善对预训练数据的引用
 - Transformer注意力权重可视化： https://github.com/mattneary/attention
 - CNN可视化：https://github.com/okdalto/CNN-visualization
 
## LLMs with security, privacy 
 - 《Toxicity in ChatGPT: Analyzing Persona-assigned Language Models》A Deshpande, V Murahari, T Rajpurohit, A Kalyan, K Narasimhan [Princeton University & The Allen Institute for AI & Georgia Tech] (2023)
 - 《Multi-step Jailbreaking Privacy Attacks on ChatGPT》H Li, D Guo, W Fan, M Xu, Y Song [Hong Kong University of Science and Technology & Peking University] (2023)
 - github.com/Cranot/chatbot-injections-exploits 收集了一些Chatbot注入和漏洞的例子，以帮助人们了解Chatbot的潜在漏洞和脆弱性
 - Large Language Models Can Be Used To Effectively Scale Spear Phishing Campaigns 探讨了大型语言模型(LLM)如何用于钓鱼攻击
   
## 资料汇总
1. 类ChatGPT模型汇总：https://github.com/chenking2020/FindTheChatGPTer
2. 类ChatGPT模型汇总（模型、数据、评估）：https://github.com/FreedomIntelligence/LLMZoo
3. 中文图像数据标注工具：https://github.com/opendatalab/labelU/blob/release/v0.5.5/README_zh-CN.md
4. 类ChatGPT模型，ChatGPT相关论文汇总：https://github.com/MLNLP-World/Awesome-LLM
5. https://blog.replit.com/llm-training: How to train your own Large Language Models
6. 基础模型相关文献资源列表：github.com/uncbiag/Awesome-Foundation-Models
7. LLaMA系列模型综述：https://agi-sphere.com/llama-models/?continueFlag=55e456ec3bc9bcf0d27e8ecc80f8e10d
8. 非官方ChatGPT资源合集：github.com/sindresorhus/awesome-chatgpt
9. https://github.com/RUCAIBox/LLMSurvey/tree/main
10. 生成式人工智能（第二版）：https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition
