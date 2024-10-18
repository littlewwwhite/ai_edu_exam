from langchain_openai import ChatOpenAI
from openai import OpenAI
from llm import get_llm
from langchain.prompts import ChatPromptTemplate

sys_prompt="""
# Role: 多项选择题的编写专家，多年执教经验的老师、命题人，懂得易错点和重点知识

## Background: 
- 我希望能够根据一段知识点文本和要求生成一套高质量的多项选择题，能够通过试题反应出学生的知识掌握水平。

## Attention:
- 优秀的多项选择题是我们评估学生对知识掌握程度的重要工具。如果题目设计不当，无法准确反映学生的实际水平，可能会导致筛选出的学生并不真正掌握学科知识。因此，请务必高度重视题目设计的质量，确保能够筛选出对学科知识掌握牢固且正确的优秀学生。

## Goals:
- 题型要多样化、命题思路要严谨、涉及的知识点要全面的单项选择题，能够准确反馈出学生对知识的真实掌握程度。
- 试题需要主题明确、知识点针对性强，错误选项需具有一定的干扰性，即看似合理但实则错误。正确答案决不允许胡编乱造。
- 给定文本，你的任务是制作一个包含5道多项选择题的深度学习学科测验
- 问题难度为：困难。

## Workflows:
1. 第一步：明确题目主题:
2. 第二步：列出至少3个与主题紧密相关的关键知识点:
3. 第三步：基于知识点文本，确保制作5道单项选择题，每道题须有4个选项，至少有两个正确答案，，确保按照RESPONSE_JSON的格式组织你的回答，并将其作为指南。RESPONSE_JSON={response_json}

## Rules:
- 第三步必须是json格式(例如：```json内容..```)，确保按照RESPONSE_JSON的格式组织你的回答
- 精心设计易错点，干扰选项需要有意义，使得干扰选项能够考验学生对知识点的掌握
- 相关性——主题的具体性和复杂性 
- 准确性——题干和正确答案必须有所依据 
- 合理性——答案是一定存在的、高质量的、在上下文中有意义的
- 清晰度——易于阅读；不混淆
- 对于不精准或模棱两可的知识，请不要加入到试题中以免引起歧义。
- 作为角色 <Role>, 严格遵守<Workflows>, 默认以中文语言完成要求。 

## Examples:
- {examples}


## Text:
你需要根据知识点文本来设计多选题，以下是知识点文本：
{text}
"""
llm= ChatOpenAI(api_key="ollama",
                         base_url="http://localhost:11434/v1/",
                         model="glm4",
                         # top_p=0.7,
                         temperature=0.98,
                        max_tokens=1000)
prompt = ChatPromptTemplate.from_messages([

    ("system", sys_prompt),

])
response_json="""{"1": {"question": "choice here", "C": "choice here", "D": "choice here"}, "correct": "correct answer", "explanation": "Detailed option explanations"}, "2": {"question": "multiple choice question", "options": {"A": "choice here", "B": "choice here", "C": "choice here", "D": "choice here"}, "correct": "correct answer", "explanation": "Detailed option explanations"}, "3": {"question": "multiple choice question", "options": {"A": "choice here", "B": "choice here", "C": "choice here", "D": "choice here"}, "correct": "correct answer", "explanation": "Detailed option explanations"}}"""
# text="""[Document(page_content='Entities:\n- {\'descriptionText\': \'图灵机是艾伦·图灵提出的一种抽象计算模型，它在可计算性问题以及人工智能发展史上具有非常重要的意义，并对计算机科学产生了深远Hebbian法则是一种关于神经网络学习的理论\', \'entityName\': \'HEBBIAN法则\'}\n- {\'descriptionText\': \'图灵奖是计算机科学领域的国际最高奖项，Marvin Minsky于1969年获得此奖项\', \'enti\'福岛邦彦提出了新知机模型\', \'entityName\': \'福岛邦彦\'}\n- {\'descriptionText\': \'B型图灵机是一种可以基于Hebbian法则进行学习的机器\', \'entityName\': \'B型图灵机\'}\n- {\'descri带卷积和子采样操作的多层神经网络由福岛邦彦提出的多层神经网络结构，采用无监督学习方式训练\', \'entityName\': \'新知机\'}\n- {\'descriptionText\': \'赫布理论是Donald Hebb提出的关于神经: \'Torch3是Lua语言的一个深度学习库，是PyTorch的前身\', \'entityName\': \'TORCH3\'}\n- {\'descriptionText\': \'图灵测试是图灵提出的一个测试计算机能否展现出智能行为的准则\', \'entityN\': \'赫布型学习是基于赫布理论的学习方法\', \'entityName\': \'赫布型学习\'}\nReports:\n- {\'rank\': 6.5, \'weight\': None, \'summaryText\': "The community is centered around the concal Networks (ANN), which is a machine learning structure simulating the human brain\'s neural network. Key entities include neurons, perceptrons, nodes, and various related concepts and individuals such as Donald Hebb and Rosenblatt. These entities are interconnected through relationships that define their roles and contributions to the field of artificial intelligence and machine learning."}\n- {\'rank\': 6.5, \'weight\': None, \'summaryText\': "The community is centered around the works of Donald Hebb, a renowned Canadian neuro-psychologist, and his contributions to the field of psychology, particularly his theories on learning and synaptic plasticity. The entities in this community are primarily theoretical concepts and professional entities directly linked to Hebb\'s work and legacy."}\n- {\'rank\': 6.5, \'weight\': None, \'summaryText\': "The community is centered around the concept of Artificial Neural Networks (ANN), which is a machine learning structure simulating the human brain\'s neural network. Key entities include neurons, perceptrons, nodes, and various related concepts and individuals such as Donald Hebb and Rosenblatt. These entities are interconnected through relationships that define their roles and contributions to the field of artificial intelligence and machine learning."}\nChunks:\n- {\'chunkText\': \'函数），因此我们可以将人工神经网络看作一个可学习的函数，并将其应用到机器学习中．理论上，只要有足够的训练数据和神（Network Capacity），这与可以被储存在网络中的信息的复杂度以及数量相关．\\n1.5.3神经网络的发展历史神经网络的发展大致经过五个阶段．\\n第一阶段：模型提出第一阶段为1943 年～1969 年，是神和数学家Walter Pitts 最早提出了一种基于简单逻辑运算的人工神经网络，这种神经网络模型称为MP 模型，至此开启了人工神经网络研究的序幕．1948 年，Alan Turing 提出了一种“B 型图灵机”．\\n“B 型tts 的学生Marvin Minsky 建造了第一台神经网络机SNARC．\\nMarvin Minsky（1927～2016），人工智能领域最重要的领导者和创新者之一，麻省理工学院人工智能实验室的创始人之一．\\n因其在人工智能拟人类感知能力的神经网络模型，称为感知器（Percep-tron），并提出了一种接近于人类学习过程（迭代、试错）的学习算法．\\n在这一时期，神经网络以其独特的结构和处理信息的方法，在许多实际应用处于长年停滞及低潮状态．\\n1969 年，Marvin Minsky 出版《感知器》一书，指出了神经网络的两个关键缺陷：一是感知器无法处理“异或”回路问题；二是当时的计算机无法支持处理大https://nndl.githu的神经网络产生质疑，并导致神经网络的研究进入了十多年的“冰河期”．\\n但在这一时期，依然有不少学者提出了很多有用的模型或算法．1974 年，哈佛大学的Paul Werbos 发明反向传播算法（BackPropagNeocognitron）[Fukushima, 1980]．\\n新知机的提出是受到了动物初级视皮层简单细胞和复杂细胞的感受野的启发．\\n但新知机并没有采用反向传播算法，而是采用了无监督学习的方式来训练，因此也没有播算法重新�\', \'chunkId\': \'0a5689249dea5fa6e3eb9ba870a10ba4\', \'frequency\': 5}\n- {\'chunkText\': \'科．\\n图1.1给出了人工智能发展史上的重要事件．\\n1940194519501955196019651970005McCulloch 和Pitts提出人工神经元网络图灵机达特茅斯会议Rosenblatt提出“感知器”推理期知识期学习期知识系统兴起专家系统兴起神经网络重新流行统计机器学习兴起（支持向量机等）深度学习的兴起然遥遥无期．\\nhttps://nndl.github.io/page_begin1.2机器学习2021 年5 月18 日61.1.2人工智能的流派目前我们对人类智能的机理依然知之甚少，还没有一个通用的理论来指导如何构建一个人工智能系统究人类智能的机理来构建一个仿生的模拟系统，而另外一些研究者则认为可以使用其他方法来实现人类的某种智能行为．\\n一个著名的例子是让机器具有飞行能力不需要模拟鸟的飞行方式，而是应该研究空气接主义和行为主义三种，其中行为主义（Actionism）主要从生物进化的角度考虑，主张从和外界环境的互动中获取智能．\\n（1）符号主义（Symbolism），又称逻辑主义、心理学派或计算机学派，是指通过作．\\n人类的认知过程可以看作符号操作过程．\\n在人工智能的推理期和知识期，符号主义的方法比较盛行，并取得了大量的成果．\\n（2）连接主义（Connectionism），又称仿生学派或生理学派，是认知经网络中的信息处理过程，而不是符号运算．\\n因此，连接主义模型的主要结构是由大量简单的信息处理单元组成的互联网络，具有非线性、分布式、并行化、局部性计算以及自适应性等特性．\\n符号主义主要模型神经网络就是一种连接主义模型．\\n随着深度学习的发展，越来越多的研究者开始关注如何融合符号主义和连接主义，建立一种高效并且具有可解释性的模型．\\n1.2机器学习机器学习（Machine LnkId\': \'8baca4b3f4003191a17a1f82b043fa52\', \'frequency\': 2}\n- {\'chunkText\': \'学家Donald Hebb 在《行为的组织》（The Organization of Behavior）一书中提出突触可塑性的基本原理，D理学家，认知心理生理学的开创者．\\n“当神经元A 的一个轴突和神经元B 很近，足以对它产生影响，并且持续地、重复地参与了对神经元B 的兴奋，那么在这两个神经元或其中之一会发生某种生长过程或新或Hebb’s Rule）．\\n如果两个神经元总是相关联地受到刺激，它们之间的突触强度增加．\\n这样的学习方法被称为赫布型学习（Hebbian learning）．\\nHebb 认为人脑有两种记忆：长期记忆和短期记忆．固作用．\\n人脑中的海马区为大脑结构凝固作用的核心区域．\\n1.5.2人工神经网络人工神经网络是为模拟人脑神经网络而设计的一种计算模型，它从结构、实现机理和功能上模拟人脑神经网络．\\n人工神另一个节点的影响大小．\\n每个节点代表一种特定函数，来自其他节点的信息经过其相应的1 图片来源：https://commons.wikimedia.org/wiki/File:Neuron_Hand-tuned.svghttps://nndl.github.io/page_begin1.5神经网络2021 年5 月18 日14权重综合计算，输入到一个激活函数中并得到一个新的活性值（兴奋或抑制）．\\n从系统观点看，人工神经元网络是由大量神经元通过极其丰富和完善的连接而构成的自学习能力．\\n首个可学习的人工神经网络是赫布网络，采用一种基于赫布规则的无监督学习方法．\\n感知器是最早的具有机器学习思想的神经网络，但其学习方法无法扩展到多层的神经网络上．\\n感知器参习问题．\\n在本书中，人工神经网络主要是作为一种映射函数，即机器学习中的模型．\\n由于人工神经网络可以用作一个通用的函数逼近器（一个两层的神经网络可以逼近任意的函数），因此我们可以将人函数的能力\', \'chunkId\': \'84a8a6e9ba478ecb40b16e75fb368359\', \'frequency\': 2}\nRelationships:\n- {\'entityFrom\': \'图灵奖\', \'entityTo\': \'MARVIN MINSKY\', \'descriptionText\vin Minsky因其在人工智能领域的贡献而获得图灵奖\', \'relation\': \'RELATED\'}\n- {\'entityFrom\': \'赫布理论\', \'entityTo\': \'DONALD HEBB\', \'descriptionText\': \'Donald Hebb提出了ELATED\'}\n- {\'entityFrom\': \'TORCH3\', \'entityTo\': \'PYTORCH\', \'descriptionText\': \'Torch3是PyTorch的前身，两者在技术继承上有关联\', \'relation\': \'RELATED\'}\n- {\'entityFr \'entityTo\': \'人工神经网络\', \'descriptionText\': \'赫布型学习是早期人工神经网络的一种学习方式赫布网络是首个可学习的人工神经网络，采用基于赫布规则的无监督学习方法\', \'relation\'工智能\', \'descriptionText\': \'图灵机对人工智能的理论基础有重要贡献\', \'relation\': \'RELATED\'}\n- {\'entityFrom\': \'福岛邦彦\', \'entityTo\': \'新知机\', \'descriptionText\': \网络的结构提供了新的视角\', \'relation\': \'RELATED\'}\n- {\'entityFrom\': \'新知机\', \'entityTo\': \'福岛邦彦\', \'descriptionText\': \'福岛邦彦提出了新知机模型福岛邦彦提出新知机模TED\'}\n- {\'entityFrom\': \'图灵机\', \'entityTo\': \'TURING TEST\', \'descriptionText\': \'图灵机模型和图灵测试都由图灵提出，对人工智能领域有深远影响\', \'relation\': \'RELATED\'}\T\', \'entityTo\': \'图灵机\', \'descriptionText\': \'图灵机模型和图灵测试都由图灵提出，对人工智能领域有深远影响\', \'relation\': \'RELATED\'}\n- {\'entityFrom\': \'赫布理论\', \'entt\': \'赫布型学习是基于赫布理论的学习方法\', \'relation\': \'RELATED\'}\n- {\'entityFrom\'"""
examples="""{
    "1": {
        "question": "关于强化学习和神经网络模型的结合使用，以下哪些描述是正确的？",
        "options": {
            "A": "强化学习可以看作是一种端到端的学习方法，其中每个内部组件直接从最终奖励中学习。",
            "B": "在深度强化学习中，贡献度分配问题指的是如何合理地将整体奖励分配给各个决策步骤。",
            "C": "误差反向传播算法主要用于监督学习，而不是强化学习。",
            "D": "强化学习中的智能体通过与环境交互来优化策略，而不需要显式的训练数据集。"
        },
        "correct": "B, D",
        "explanation": "B. 贡献度分配问题是深度强化学习中的一个关键问题，涉及如何将整体奖励分配给各个决策步骤。D. 强化学习的核心是通过与环境交互来学习最优策略，而不依赖于预先标记的数据集。A. 选项描述不准确，强化学习通常不是纯粹的端到端学习；C. 反向传播算法确实主要应用于监督学习，但在某些情况下也可用于强化学习中的策略梯度方法。"
    },
    "2": {
        "question": "下列关于神经元及其在人工神经网络中的模拟，哪几项是正确的？",
        "options": {
            "A": "神经元之间的连接强度（突触权重）在人工神经网络中是固定不变的。",
            "B": "Hebbian理论认为，当两个神经元同时被激活时，它们之间的连接会变得更强。",
            "C": "反向传播算法是由Paul Werbos提出的，它解决了多层神经网络中贡献度分配的问题。",
            "D": "在生物神经元中，树突负责接收来自其他神经元的信息，轴突则负责传递信息至下一个神经元或效应器。"
        },
        "correct": "B, C, D",
        "explanation": "B. Hebbian理论强调了神经元之间连接强度的可塑性，这是学习的基础。C. Paul Werbos确实在1974年提出了反向传播算法，该算法对于解决多层神经网络中的权重更新问题至关重要。D. 树突和轴突的功能正如所述。A. 神经网络中的突触权重是通过学习过程不断调整的，并非固定不变。n在这一时期，神经网络以其独特的结构和处理信息的方法，在许多实际应用处于长年停滞及低潮状态．\\n1969 年，Marvin Minsky 出版《感知器》一书，指出了神经网络的两个关键缺陷：一是感知器无法处理“异或”回路问题；二是当时的计算机无法支持处理大https://nndl.githu的神经网络产生质疑，并导致神经网络的研究进入了十多年的“冰河期”"
    },
    "3": {
        "question": "关于特征表示方法以及其在机器学习中的应用，下列说法正确的是？",
        "options": {
            "A": "one-hot编码适合于表示连续变量。",
            "B": "局部分表示如one-hot向量，通常用于表示离散特征，例如颜色。",
            "C": "分布式表示相较于one-hot编码，能更好地捕捉特征间的相似性。",
            "D": "嵌入（Embedding）技术仅适用于自然语言处理领域。"
        },
        "correct": "B, C",
        "explanation": "B. one-hot向量常用来表示离散特征，比如不同的颜色。C. 分布式表示能够表达特征间的语义关系，这在one-hot编码中是无法实现的。A. one-hot编码不适合表示连续变量，因为它是为离散变量设计的。D. 嵌入技术不仅限于自然语言处理，也广泛应用于推荐系统等领域。"
    }
}"""
# Set up the language model and memory chain
text="""图灵机是艾伦·图灵提出的一种抽象计算模型，它在可计算性问题以及人工智能发展史上具有非常重要的意义，并对计算机科学产生了深远Hebbian法则是一种关于神经网络学习的理论\', \'entityName\': \'HEBBIAN法则\'}\n- {\'descriptionText\': \'图灵奖是计算机科学领域的国际最高奖项，Marvin Minsky于1969年获得此奖项\',．\\n因其在人工智能拟人类感知能力的神经网络模型，称为感知器（Percep-tron），并提出了一种接近于人类学习过程（迭代、试错）的学习算法．\\n在这一时期，神经网络以其独特的结构和处理信息的方法，在许多实际应用处于长年停滞及低潮状态．\\n1969 年，Marvin Minsky 出版《感知器》一书，指出了神经网络的两个关键缺陷：一是感知器无法处理“异或”回路问题；二是当时的计算机无法支持处理大https://nndl.githu的神经网络产生质疑，并导致神经网络的研究进入了十多年的“冰河期”．\\n但在这一时期，依然有不少学者提出了很多有用的模型或算法．1974 年，哈佛大学的Paul Werbos 发明反向传播算法（BackPropagNeocognitron）[Fukushima, 1980]．\\n新知机的提出是受到了动物初级视皮层简单细胞和复杂细胞的感受野的启发． '赫布理论是Donald Hebb提出的关于神经: \'Torch3是Lua语言的一个深度学习库，是PyTorch的前身\', \'entit"""
chain = prompt | llm
user_input="请开始出题："
result=chain.invoke({"question":user_input,"examples": examples,"text":text,"response_json":response_json})
print(result.content)

print(result.usage_metadata)
