# Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap

<div align="center">
<a href="https://arxiv.org/abs/2503.21411"><img src="https://img.shields.io/badge/arXiv-2503.21411-b31b1b.svg" alt="arXiv Badge"/></a>
<a href="https://github.com/tongnie/awesome-llm4tr/stargazers"><img src="https://img.shields.io/github/stars/tongnie/awesome-llm4tr" alt="Stars Badge"/></a>
</div>

<p align="center">
<img src="Assets/Fig1.png" width="580"/>
</p>
<h5 align=center>The LLM4TR framework proposed in this survey.</h5>


> A collection of papers and resources related to **Large Language Models for Transportation (LLM4TR)**. 
>
> This is the online content of our survey [**"Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap"**]([https://arxiv.org/abs/2503.21411](https://arxiv.org/abs/2503.21411)). [![Paper page](https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm-dark.svg)](https://arxiv.org/abs/2503.21411)
>
> 📢 This project will continue to be updated, stay tuned!
> 
> Feel free to contact us if you have any suggestions or would like to discuss with us by e-mail: tong.nie@connect.polyu.hk, wei.w.ma@polyu.edu.hk
>
>
> 🤝 If you find our survey or repository useful for your research, please cite the following paper:

```
@article{nie2025LLM4TR,
    title={Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap},
    author={Tong Nie and Jian Sun and Wei Ma},
    year={2025},
    journal={arXiv preprint arXiv:2503.21411},
    url={https://arxiv.org/abs/2503.21411}
}
```

## :fire: Update
- 📝 TODO: Update the repository of open-source projects.
- 📝 TODO: Update the papers of each category.
- ✅ Update the list of resources.
- ✅ Initialize the repository.
- ✅ [March 2025] ArXiv Version: Our paper has been released in arXiv!


## :page_with_curl: Framework and Taxonomy
*Definition*: LLM4TR refers to the methodological paradigm that systematically harnesses emergent capabilities of LLMs to enhance transportation tasks through four synergistic roles: transforming raw data into understandable insights, distilling domain-specific knowledge into computable structures, synthesizing adaptive system components, and orchestrating optimal decisions. We survey the existing literature and summarize how LLMs are exploited to solve transportation problems from a methodological perspective, i.e., **the roles of LLMs in transportation systems**. They generally include four aspects:

<p align="center">
<img src="Assets/Fig2.png" width="680"/>
</p>
<h5 align=center>The literature classification procedure in this survey.</h5>

- **LLMs as information processors**
    - *Function*: LLMs process and fuse heterogeneous transportation data from multiple sources (text, sensor data, task description, and user feedback) through contextual     encoding, analytical reasoning, and multimodal integration. They enable unified processing of complex traffic patterns, parsing and integrating multi-source information to assist in the managing and semantic understanding of traffic data, reducing the complexity of downstream tasks.
- **LLMs as knowledge encoders**
    - *Function*: LLMs extract and formalize transportation domain knowledge from unstructured data through explicit rule extraction and latent semantic embedding. This role bridges the gap between the unstructured domain knowledge inherent in the data and computable (or comprehensible) representations for downstream applications.
- **LLMs as component generators**
    - *Function*: LLMs create functional algorithms, synthetic environments, and evaluation frameworks through instruction-followed content generation. This role utilizes generative capabilities of LLMs to automate the design, testing, and refinement of components in intelligent transportation systems.
- **LLMs as decision facilitators**
    - *Function*: LLMs predict traffic dynamics, optimize decisions, and simulate human-like reasoning, establishing new paradigms as generalized task solvers. This role employs LLMs as predictive engines and decision facilitators for both micro-level agent behaviors and macro-level system states.

## 📈 Research trend
As an intuitive overview of the current research trend and focus, we visualize the statistics of selected papers according to our taxonomy.

<p align="center">
<img src="Assets/Fig3.png" width="680"/>
</p>
<h5 align=center>Heatmap of the current research trend and pie chart of the proportion of the four roles of LLMs in different tasks.</h5>


## ⭐ Overview of mainstream LLMs
| **Model**                        | **Release Date**    | **Organization** | **Size (B)**   | **Data (TB**  | **Hardware Cost**    | **Public Access** |
|---------------------------------------|---------------------|-----------------------|-----------------|----------------|----------------------|-----------------|
| T5       | 2019.10             | Google                | 11              | 750 GB of text | 1024 TPU v3          | Yes             |
| GPT-3     | 2020.5              | OpenAI                | 175             | 300 B tokens   | -                    | No              |
| PaLM       | 2022.4              | Google                | 540             | 780 B tokens   | 6144 TPU v4          | No              |
| LLaMA         | 2023.2              | Meta                  | 65              | 1.4 T tokens   | 2048    A100 GPU   |       Partial  | 
| GPT-4           | 2023.3              | OpenAI                | -               | -              | -                    | No              |
| LLaMA-2      | 2023.7              | Meta                  | 70              | 2 T tokens     | 2000    A100 GPU              | Yes                 |
| Mistral-7B | 2023.9              | Mistral AI            | 7               | -              | -                    | Yes             |
| Qwen-72B          | 2023.11             | Alibaba               | 72              | 3 T tokens     | -                    | Yes             |
| Grok-1                                | 2024.3              | xAI                   | 314             | -              | -                    | Yes             |
| Claude 3                              | 2024.3              | Anthropic             | -               | -              | -                    | No              |
| GLM-4-9B        | 2024.6              | Zhipu AI              | 9               | 10 T tokens    | -                    | Yes             |
| LLaMA-3.1       | 2024.7              | Meta                  | 405             | 15 T tokens    | 16 thousand H100 GPU | Yes             |
| Gemma-2         | 2024.6              | Google                | 27              | 13 T tokens    | 6144 TPUv5p          | Yes             |
| DeepSeek-V3    | 2024.12             | DeepSeek              | 671 | 14.8 T tokens  | 2048 H800 GPU        | Yes             |


## 📋 Summary of language-enhanced ITS and autonomous driving datasets for LLM development and evaluation benchmarks
| **Dataset**                                             | **Year**                                                                 | **Venue** | **Task**                                                                                      | **Use Case in LLM Development**                                                  |
|--------------------------------------------------------------|-------------------------------------------------------------------------------|----------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset)                                | 2018                                                                          | ECCV           | Action interpretation and control signal prediction                                                | Explainable end-to-end driving through visual question answering.                     |
| [SUTD-TrafficQA](https://github.com/sutdcv/SUTD-TrafficQA)                           | 2021                                                                          | CVPR           | Video causal reasoning over traffic events                                                         | Evaluating the reasoning capability over 6 tasks.                                     |
| [TrafficSafety-2K](https://github.com/ozheng1993/TrafficSafetyGPT?tab=readme-ov-file)       | 2023                                                                          | arXiv          | Annotated traffic incident and crash report analysis                                               | GPT fine-tuning for safety situational awareness.                                     |
| [NuPrompt](https://github.com/wudongming97/Prompt4Driving?tab=readme-ov-file)                              | 2023                                                                          | AAAI           | Object-centric language prompt set for 3D driving scenes                                           | Prompt-based driving task to predict the described object trajectory.                 |
| [LaMPilot](https://github.com/PurdueDigitalTwin/LaMPilot)                               | 2024                                                                          | CVPR           | Code generation for autonomous driving decisions                                                   | CoT reasoning and instruction following for lane changes and speed adaptation.        |
| [CoVLA](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset)                                  | 2024                                                                          | arXiv          | Vision-Language-Action alignment (80+ hrs driving videos)                                          | Trajectory planning with natural language maneuver descriptions.                      |
| [VLAAD](https://github.com/sungyeonparkk/vision-assistant-for-driving)                                  | 2024                                                                          | WACV           | Natural language description of driving scenarios                                                  | QA systems for driving situation understanding.                                       |
| [CrashLLM](https://crashllm.github.io/)                            | 2024                                                                          | arXiv          | Crash outcome prediction (severity, injuries)                                                      | What-if causal analysis for traffic safety using 19k crash reports.                   |
| [TransportBench](https://agi4engineering.github.io/TransportBench/)                  | 2024                                                                          | arXiv          | Answering undergraduate-level transportation engineering problem                                   | Benchmarking LLMs on planning, design, management, and control questions.             |
| [Driving QA](https://github.com/wayveai/Driving-with-LLMs)                           | 2024                                                                          | ICRA           | 160k driving QA pairs with control commands                                                        | Interpreting scenarios, answering questions, and decision-making.                     |
| [MAPLM](https://github.com/LLVM-AD/MAPLM)                                   | 2024                                                                          | CVPR           | Multimodal traffic scene dataset including context, image, point cloud, and HD map                 | Visual instruction-tuning LLMs and VLMs and vision QA tasks.                          |
| [DrivingDojo](https://github.com/Robertwyq/Drivingdojo)                     | 2024                                                                          | NeurIPS        | Video clips with maneuvers, multi-agent interplay, and driving knowledge | Training and action instruction following benchmark for driving world models. |
| [TransportationGames](https://arxiv.org/abs/2401.04471)   | 2024                                                                          | arXiv          | Benchmarks of LLMs in memorizing, understanding, and applying transportation knowledge on 10 tasks | Grounding (M)LLMs in transportation-related tasks.                                    |
| [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA)                         | 2024                                                                          | AAAI           | Benchmark for vision QA in autonomous driving, including 34K visual scenes and 460K QA pairs       | Developing 3D detection and VQA techniques for end-to-end autonomous driving systems. |
| [TUMTraffic-VideoQA](https://traffix-videoqa.github.io/)                | 2025                                                                          | aXiv           | Temporal traffic video understanding                                                               | Benchmarking video reasoning for multiple-choice video question answering.            |
| [V2V-QA](https://eddyhkchiu.github.io/v2vllm.github.io/)                                  | 2025                                                                          | arXiv          | Cooperative perception via V2V communication                                                       | Fuse perception information from multiple CAVs and answer driving-related questions.  |
| [DriveBench](https://drive-bench.github.io/)                               | 2025                                                                          | arXiv          | A comprehensive benchmark of VLMs for perception, prediction, planning, and explanation            | Visual grounding and multi-modal understanding for autonomous driving.                |

## 📄 Representative surveys on LLMs and related techniques
| **Paper Title**                                                                        | **Year** | **Venue** | **Scope and Focus**                                                                                                                             |
|---------------------------------------------------------------------------------------------|---------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| [A survey of Large Language Models](https://arxiv.org/abs/2303.18223)                                    | 2023          | arXiv          | Reviews the evolution of LLMs, pretraining, adaptation, post-training, evaluation, and benchmarks.                                                   |
| [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)                 | 2024          | arXiv          | Reviews LLM families (GPT, LLaMA, PaLM), training techniques, datasets, and benchmark performance.                                                   |
| [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)  | 2023          | arXiv          | Introduces the progress of RAG paradigms, including the naive RAG, the advanced RAG, and the modular RAG.                                            |
| [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234)                                   | 2022          | arXiv          | Summarizes training strategies, prompt designing strategies, and various ICL application scenarios, such as data engineering and knowledge updating. |
| [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/abs/2308.10792)         | 2023          | arXiv          | Reviews methodology of SFT, SFT datasets, applications to different modalities, and influence factors.                                               |
| [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/abs/2212.10403)               | 2022          | ACL            | Examines techniques for improving and eliciting reasoning in LLMs, methods and benchmarks for evaluating reasoning abilities.                        |
| [A Survey of LLM Surveys](https://github.com/NiuTrans/ABigSurveyOfLLMs)               | 2024          | GitHub         | Compiles 150+ surveys across subfields like alignment, safety, and multimodal LLMs.                                                                  |



## 📖 Popular open-source libraries for LLM development
| **Library Name**     | **Basic Functions**                                                                       | **Use Cases**                                                             | 
|---------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Pretrained models (NLP, vision) and fine-tuning pipelines                                      | Model deployment, adapt tuning                                                 | 
| [DeepEval](https://github.com/confident-ai/deepeval)                  | Framework for evaluating LLM outputs using metrics like groundedness and bias                  | Educational applications, hallucination detection                              |
| [RAGAS](https://github.com/explodinggradients/ragas)                     | Quantifies RAG pipeline performance                                                            | Context relevance scoring, answer quality                                      | 
| [Sentence Transformers](https://www.sbert.net/)    | Generates dense text embeddings for semantic similarity tasks                                  | Survey item correlation analysis, retrieval                                    | 
| [LangChain](https://www.langchain.com/)                 | Chains LLM calls with external tools for multi-step workflows                                  | RAG, agentic reasoning, data preprocessing                                     |
| [DeepSpeed](https://www.deepspeed.ai/)                 | A deep learning optimization library developed by Microsoft, which has been used to train LLMs | Distributed training, memory optimization, pipeline parallelism                | 
| [FastMoE](https://fastmoe.ai/)                   | A specialized training library for MoE models based on PyTorch                                 | Transfer Transformer models to MoE models, data parallelism, model parallelism | 
| [Ollama](https://ollama.ai)                    | Local LLM serving with support for models like Llama and Mistral                               | Offline inference, privacy-sensitive apps                                      | 
| [OpenLLM](https://github.com/bentoml/OpenLLM)                  | Optimizes LLM deployment as production APIs compatible with OpenAI standards                   | Scalable model serving, cloud/on-prem hosting                                  | 

## 🖥️ A rouge estimate of the hardware requirements for fine-tuning and inference across LLaMA model sizes. 
> BS = Batch Size. Estimated values marked "(est.)" derive from scaling laws. Inference rates measured at batch size 1 unless noted. Actual requirements and performance may differ for specific configurations.

| **Model Size** | **Full Tuning GPUs**  | **LoRA Tuning GPUs**  | **Full Tuning BS/GPU** | **LoRA BS/GPU** | **Tuning Time (Hours)** | **Inference Rate (Tokens/s)** |
|---------------------|----------------------------|----------------------------|-----------------------------|----------------------|------------------------------|------------------------------------|
| 7B         | 2×A100 80GB         | 1×RTX 4090 24GB     | 1-2                         | 4-8                  | 3-5                          | 27-30                              |
| 13B        | 4×A100 80GB (est.)  | 2×A100 40GB         | 1                           | 2-4                  | 8-12                         | 18-22                              |
| 70B        | 8×H200 80GB         | 4×H200 80GB         | 1                           | 1-2                  | 24-36                        | 12-15                              |
| 405B      | 64×H200 80GB (est.) | 16×H200 80GB (est.) | 1 (est.)                    | 1 (est.)             | 72-96 (est.)                 | 5-8                                |



## License

This repository is released under the [MIT LICENSE](LICENSE).
