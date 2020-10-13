## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

#### Google AI Language团队



## 摘要

我们介绍一个新的语言表达模型叫做BERT，它表示Bidirectional Encoder Representations from Transformers。不像最近的语言模型(Peters et al., 2018a; Radford et al., 2018)，BERT可以通过在左边和右边上下文中共同调节，从而在未标注的文本中预训练深度双向表达。结果，在预训练BERT模型上，仅仅增加一个额外的输出层便可以在许多任务中达到state-of-the-art效果，比如问答任务和语言推理任务，这不会因为具体任务的不同而改变模型的本质结构。

BERT概念简单且效果好。它在十一个语言处理任务中达到了state-of-the-art效果，比如将GLUE分数提升至了80.5%（7.7%的绝对提升）；将MiltiNLI正确率提升至了86.7%（4.6%的绝对提升）；SQuAD v1.1问答测试Test F1提升至了93.2（1.5的绝对提升）；SQuAD v2.0问答测试Test F1提升至了83.1（5.1的绝对提升）；

### 1 介绍

语言模型的预训练在提高自然语言处理任务中已经展示出了一定效果。这些包括sentence-level 任务比如语言推理和释义（这些任务都是通过整体分析来预测句子之间的关系），以及token-level 任务比如命名实体识别和问答，在这里模型都需要在token level上有细致的输出

将预训练语言表达模型应用在downstream task中有两种现存的方法：基于功能feature-based和微调fine-tuning。feature-based方法，比如ELMo，包括了预训练表达作为额外特征，使用了任务特定的结构。fine-turing方法，比如Generative Pre-trained Transformer (OpenAI GPT) ，引入了最小特定任务参数，它是通过微调所有预训练参数基于downstream tasks训练的。这两种方法在预训练时共享相同的目标函数，他们使用单向语言模型来学习总体语言表达。

我们任务现在的技术限制了预训练表示的能力，特别在微调表示方面。标准语言模型的主要显示是模型是单向的，这限制了在预训练中模型结构的选择。比如说，在OpenAI GPT，作者使用了从左到右结构，在Transforme中self-attention层中，这里每一个token只能关注前一个token。对sentence-level任务来说，这种限制是次优的，并且在基于微调方法时的token-level任务比如问题任务中，这可能是有害的。包含两个方向的语境很关键。

在这篇论文中，通过使用BERT，我们改进了基于微调的方法。BERT使用了掩盖语言模型（“masked language model” ，MLM，由Cloze task (Taylor, 1953)提出）减轻了单向限制。MLM随机掩盖输入中的一些token，它的目标是预测基于语境中被掩盖原来单词的id。不像从左到右语言预训练模型，MLM的目标使得表示可以融合左边和右边内容，这让我们预训练一个深度双向转换器deep bidirectional Transformer。除了使用MLM之外，我们也使用后句预测任务，共同预训练文本对表示。我们论文做出了以下贡献。

- 我们解释了语言表示中双向预训练（bidirectional pre-training for language representations）的重要性。不像like Radford et al. (2018)在预训练中使用单向语言模型，BERT在预训练中使用MLM。这可以与Peters et al. (2018a)对比，他们简单地独立地连接了从左到右和从右到左的语言模型。
- 我们证明，预训练模型减少了许多繁琐工程上特定任务的结构。BERT是第一个基于微调的表达模型并且在大量的sentence-level和token-level任务中达到了state-of-the-art表现，超出了很多特定任务的模型。
- BERT在11个NLP任务中提高了state-of-the-art水平。代码和预训练模型在https://github.com/google-research/bert公开发布。

### 2 相关工作

预训练语言模型已经有很长的历史了，在本节中我们简述一些最普遍用到的方法。

#### 2.1 基于特征方法的无监督学习 Unsupervised Feature-based Approaches

#### 2.1 Unsupervised Feature-based Approaches

#### 2.2 基于微调方法的无监督学习 Unsupervised Fine-tuning Approaches

#### 2.3 监督数据中的迁移学习 Transfer Learning from Supervised Data

### 3 BERT

这节我们将介绍BERT和它具体的实现。在我们的框架中有两个步骤：预训练和微调。在预训练中，模型是基于在不同预训练目标中未标注的数据。对于微调，BERT模型先以预训练参数初始化，并且所有的参数都基于下行数据进行微调。每个下行数据任务有不同的微调模型，它们甚至都以相同的预训练参数初始化。将以图表1中的一个问答例子作为这节的例子。

BERT的一个独特特点是它在不同的任务中都可以用相同的结构。在预训练结构和最终下行结构中只有很小的区别。

**模型结构** BERT的模型结构是一个多层双向Transformer编码模型，它基于Vaswani et al. (2017)中最初的实现并且在tensor2tensor library中发布。Transformer模型的使用已经很常见并且我们的实现与其原版几乎差不多，我们将省略一个模型结构中描述的详细的背景，请读者参最好的指导“The Annotated Transformer.” Vaswani et al. (2017) 。

对于此任务，我们记L为模型层数，H为隐藏尺寸，A为self-attention heads。我们首先介绍在两个不同模型尺寸上的结果：

$$
BERT_{BASE}(L=12,H=768,A=12,Total Parameters=110M)
$$

$$
BERT_{LARGE}(L=24,H=1024,A=16,Total Parameters=340M)
$$

BERT<sub>BASE</sub>与OpenAL GPT相比，有相同的模型尺寸。但重要的是，BERT Transformer使用双向自注意力，而GPT Transformer使用了有限的自注意力以至于每个token只能关注它左边的内容。

**输入输出表示** 为了能让BERT处理多种down-stream tasks，我们的输入表示能够在个token序列中清楚地展示一个单句和一组句子（例子，<问题，回答> ）。通过这项工作，一个句子可以被连续的文本以任意跨度解释，而不是一个真实的语言句子。一个”序列“代表了BERT中的输入token序列，它也有可能是一个单句或者两个句子的组合。

我们通过30,000个token词语进行词块嵌入(Wu et al., 2016)。每个句子最开始的token总是一个special classification token ([CLS])。最终对应着隐藏状态与此标记对应的用于分类任务的聚合序列表示相符合。句子对被包装成一个单句。我们通过两种方法区别这些句子。第一步，我们用一个特殊的token([SEP])将他们分离。第二步，我们为每一个token添加一个learned embedding表示它是属于句子A还是句子B。正如图表1中所示，我们将input embedding记作E，特殊token([SEP])的最终隐藏向量记作$C\in R^H$，对于$i^{th}$输入token，有最终隐藏向量记为$T_i\in R^H$。

![图片1](https://img.vim-cn.com/79/3a0df9b31af62bc6a4439e08ce07b76b37ec71.png)

图表1：BERT中总体预训练和微调过程。除输入层之外，相同的结构也在预训练和微调中使用。同样的预训练模型参数也被用来初始化不同的downstream tasks。在微调过程中，所有的参数都被微调。[CLS]是在每一个输入样例前加入的特殊标记(e.g. separating questions/answers)。

对一个给定的token，它的输入表示通过统计对应的token，segment和 position embeddings的合计来构建的。这种可视化结构在图表2中表示。

![](https://img.vim-cn.com/11/d43b94ac4fa101646bd145d46d5394064b7c89.png)

图表2：BERT输入表示。 input embeddings是 token embeddings、the segmentation embeddings、 the position embeddings的总和。

### 3.1 预训练BERT

不像Peters et al. (2018a) and Radford et al. (2018)，我们不用传统的从左到右或从右到左的语言模型来预训练BERT。这节中将讲到，我们预训练BERT模型用了两个无监督任务。这一步在图表1的左半部分有解释。

#### 任务一：Masked LM（完形填空）

直观来看，我们有理由相信，深度双向模型的确比  从左到右模型  或  简单的把从左到右模型和从右到左模型相连接  效果更好。不幸的是，传统标准语言模型只能从左到右或者从右到左训练，尽管双向情况会让每个词不直接地“看见它自己”，并且模型只能简单地在一个多层语境中预测目标词。

为了训练一个深度双向表示，我们仅仅按一定比例随机地抹去了一些输入token，然后预测这些掩盖的tokens。我们将这个过程叫做““masked LM” (MLM)”，尽管它在文学中(Taylor, 1953)经常代表Cloze task。在这种情况下，最终隐藏向量和mask tokens被送入在词语上的output softmax。我们所有的实验都在每个序列中随机掩盖了15%的tokens。和降噪自动编码(Vincent et al., 2008)相比，我们只预测掩盖的词而不是重建整个输入。

尽管这让我们可以得到一个双向预训练模型，但其中的一个缺点是我们在预训练和微调中产生了错误匹配。这是因为[mask] token在微调时候并不出现。为了减轻这点，我们并不总是将“masked” words替换为正确的[mask] token。训练数据产生机会随机挑选15%的token位置来预测。如果第i个token被选中，我们将对第i个token执行以下替换策略：

1. 80%的概率：替换为[MASK] token
2. 10%的概率：替换为随机token
3. 10%的概率：不改变第i个token

之后，$T_i$将通过交叉熵损失来预测原始的token。我们在附录C.2中对比了这个过程不同变式。

<img src="https://img.vim-cn.com/f2/252ce9e052fd89f23763cf92ad39db123adefa.png" style="zoom: 50%;" />

#### 任务二：Next Sentence Prediction（后句预测）

很多重要的downstream tasks比如问答（QA）和自然语言引用（NLI）都是在理解两个句子间关系的基础上的，这种关系不能直接被语言模型获取到，我们预训练了一个二值后句预测任务，它可以简单地从任何单一语言语料库中生成。特别地，当为每个训练样本选择句子A和B时，有50%的概率句子B是正确的跟在A后的句子（以IsNext标记），还有50%的概率它是一个从语料库随机取出的句子（以NotNext标记）。正如我们再图表1中所示，C用来预测后句。尽管它很简单，我们在5.1节中展示了这种预训练是对QA和NLI都非常有益。NSP任务和representation learning objectives紧密相联系。但是，在先前的工作中，BERT为了初始化结束任务模型参数而变化了所有的参数。

**预训练数据** 预训练过程主要遵循现存的语言模型预训练文献。我们使用BooksCorpus (800M words)和English Wikipedia (2,500M words)作为预训练的语料库。对于维基百科，我们只抽取文本文章并且忽略列表、表格和页眉。为了抽取又长有连续的句子，使用文档级语料库而不改组句子级别的语料库是很危险的。

### 3.2 微调BERT

微调是非常简单的因为在 Transformer中的self-attention机制可以使BERT为许多downstream tasks建模，不管它们包括了单个文本或者一对，通过替换争取的输入和输出都可以做到。对于涉及到文本对的应用，一个常见的模式是在应用bidirectional cross attention中为文本对独立编码，比如Parikh et al. (2016); Seo et al. (2017)。BERT使用self-attention机制来统一这两个阶段，在编码具有self-attention的文本时，它有效地包括了两个句子之间的bidirectional cross attention。

对于每个任务，我们会将特定任务的输入和输出插入到BERT中，并且从头到尾微调所有的参数，在输入中，预训练中的句子A和句子B都和以下内容相像：①释义中的句子对，②隐含中的假设——前提对，③问答中的问题——文章对，还有④文本分类和序列标签中退化的文本——空集对。在输出中，token representations被嵌入一个token-level 任务的输出层，比如序列标定和问题回答，并且在文本分类中[CLS]表达式也被输入到一个输出层中，比如隐含意义或者情感分析。

与预训练相比，微调相对便宜。从一个精确一样的预训练模型开始，文中的所有结果都可以在一个谷歌云TPU上最多花一个小时复现出来，或者再GPU上花几个小时也可以。我们在第四节的相关描述中说明具体任务的细节。更多的细节都可以再附录A.5中找到。



### 4 实验

就是介绍很多具体的实验...



记录一下生词

**Token-level**：Token-level classification means that each token will be given a label, for example a part-of-speech tagger will classify each word as one particular part of speech. Each token (element in the sequence) will have a corresponding label in the output.

**downstream tasks（下游任务）**: Downstream tasks is what the field calls those supervised-learning tasks that utilize a pre-trained model or component

**feature-based**

**fine tuning（微调）**：精细地调节模型参数以至于拟合某种观测结果。

**state-of-the-art（顶尖水平效果）**：代表某一时期在技术科学领域的最高水平

**MLM(masked language models)**

**self-attention（自注意力）**

**learned embedding**

**hidden vector**

**Ablation Studies**