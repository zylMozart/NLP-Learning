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

#### 任务一：Masked LM

直观来看，我们有理由相信，深度双向模型的确比  从左到右模型  或  简单的把从左到右模型和从右到左模型相连接  效果更好。不幸的是，传统标准语言模型只能从左到右或者从右到左训练，尽管双向情况会让每个词不直接地“看见它自己”，并且模型只能简单地在一个多层语境中预测目标词。

为了训练一个深度双向表示，我们仅仅按一定比例随机地抹去了一些输入token，然后预测这些掩盖的tokens。我们将这个过程叫做““masked LM” (MLM)”，尽管它在文学中(Taylor, 1953)经常代表Cloze task。在这种情况下，最终隐藏向量和mask tokens被送入在词语上的output softmax。我们所有的实验都在每个序列中随机掩盖了15%的tokens。和降噪自动编码(Vincent et al., 2008)相比，我们只预测掩盖的词而不是重建整个输入。

尽管这让我们可以得到一个双向预训练模型，但其中的一个缺点是我们在预训练和微调中产生了错误匹配。这是因为[mask] token在微调时候并不出现。为了减轻这点，我们并不总是将“masked” words替换为正确的[mask] token。训练数据产生机会随机挑选15%的token位置来预测。如果第i个token被选中，我们将对第i个token执行以下替换策略：

1. 80%的概率：替换为[MASK] token
2. 10%的概率：替换为随机token
3. 10%的概率：不改变第i个token

之后，$T_i$将通过交叉熵损失来预测原始的token。我们在附录C.2中对比了这个过程不同变式。

<img src="https://img.vim-cn.com/f2/252ce9e052fd89f23763cf92ad39db123adefa.png" style="zoom: 50%;" />

记录一下

downstream task

fine tuning

state-of-the-art

self-attention

special classification token ([CLS])

learned embedding

hidden vector
