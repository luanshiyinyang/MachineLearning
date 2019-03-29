# FP-Growth
- 简介
	- FP-Growth算法是一种发现数据集中频繁模式的有效方法，它在Apriori算法的原理的基础上，采用FP（Frequent Pattern，频繁模式）树数据结构对原始数据进行压缩，大大加快了计算速度。FP-Growth算法把数据集中的事物映射到一棵FP-Tree上，再根据这棵树找到频繁项集，FP-Tree的构建过程只需要扫描两次数据集，特别是在大型数据集上具有很高的效率。
- 原理
	- FP-Growth算法的基本过程分为两个步骤：构建FP树和挖掘频繁项集。FP树构建通过两次数据扫描，将原始数据中的事物压缩到一个FP树，该FP树类似于前缀树，相同前缀的路径可以共用，从而达到压缩数据的目的。接着通过FP树找出每个项的条件模式基、条件FP树，递归的挖掘条件FP树得到所有的频繁项集。算法的主要计算“瓶颈”在FP-Tree的递归挖掘上，这里介绍FP-Growth算法主要步骤。
	- **FP树的数据结构**
		- FP-Growth算法将数据存储在一种称为FP树的紧凑数据结构中。一棵FP树看上去与计算机科学中的其他树的结构类似，但是它通过链接来连接相似元素，被连起来的元素项可以看出一个链表。
		- 与搜索树不同的是，一个元素项可以在一棵FP树中出现多次。FP树会存储项集的出现频率，而每个项集会以路径的方式存储在树中。存在相似元素的集合会共享树的一部分，只有当集合之间完全不同时，树才会分叉。树节点上给出集合中的单个元素及其在序列中的出现次数，路径会给出该序列的出现次数。
	- **构建FP树**
		- FP通过链接来连接相似元素，被连起来的元素可以看做一个链表。将事物数据表中的各个事物对应的数据项按照支持度排序后，把每个事物中的数据项按降序依次插入一棵以NULL为根节点的树中，同时在每个节点处记录该节点出现的支持度。构建FP树需要两次扫描数据集，第一次用来统计各元素项的出现频率，第二次扫描只考虑频繁项集，FP树具体构建过程如下。
			- 1.遍历数据集，统计各元素项出现次数，创建头指针表。
			- 2.移除头指针表中不满足最小值尺度的元素项。
			- 3.第二次遍历数据集，创建FP树。对每个数据集中的项集进行如下操作。
				- a.初始化空FP树。
				- b.对每个项集进行过滤和重排序。
				- c.使用这个项集更新FP树，从FP树的根节点开始进行。
					- 如果当前项集的第一个元素项存在于FP树当前节点的子节点中，则更新这个子节点的计数值。
					- 否则，创建新的子节点，更新头指针表。
					- 对当前项集的其余元素项和当前元素项的对应子节点递归c过程。
	- **从FP树中挖掘频繁项集**
		- 有了FP树，就可以抽取频繁项集了。这里的思路与Apriori算法大致类似，首先从单元素项集合开始，然后在此基础上逐步构建更大的集合。从FP树中抽取频繁项集的基本步骤如下。
			- 1.从FP树中获得条件模式基
				- 从头指针表最下面的频繁元素项开始，构造每个元素项的条件模式基。条件模式基是以所查找元素项为结尾的路径集合，这里每一条路径都是该元素项的前缀路径。条件模式基的频繁度为该路径上该元素项的频繁度计数。
			- 2.利用条件模式基，构建一个条件FP树
				- 对于每一个频繁项，都需要创建一棵条件FP树。使用刚才创建的条件模式基作为输入，累加每个条件模式基上的元素项频繁度，过滤低于阈值的元素项，采用同样的建树代码构建FP树。递归发现频繁项、条件模式基和另外的条件树。
			- 3.迭代重复步骤1和2，直到树包含一个元素项，这样就获得了所有的频繁项集。
- 实战
	- 使用FP-Growth算法提取频繁项集
	- 提取文本事物数据的频繁项集
	- 代码
		- ```python
			class treeNode:
			    """
			    定义FP树数据结构
			    """
			    def __init__(self, nameValue, numOccur, parentNode):
			        # 节点元素名称
			        self.name = nameValue
			        # 出现次数
			        self.count = numOccur
			        # 指向下一个相似节点
			        self.nodeLink = None
			        # 指向父节点
			        self.parent = parentNode
			        # 指向子节点，子节点元素名称为键，指向子节点指针为值
			        self.children = {}
			
			    def inc(self, numOccur):
			        """
			        增加节点的出现次数
			        :param numOccur:
			        :return:
			        """
			        self.count += numOccur
			
			    def disp(self, ind=1):
			        """
			        输出节点和子节点的FP树结构
			        :param ind:
			        :return:
			        """
			        print('  ' * ind, self.name, ' ', self.count)
			        for child in self.children.values():
			            child.disp(ind + 1)
			
			
			def createTree(dataSet, minSup=1):
			    """
			    建树
			    :param dataSet:
			    :param minSup:
			    :return:
			    """
			    headerTable = {}
			    for trans in dataSet:
			        for item in trans:
			            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
			
			    for k in list(headerTable):
			        if headerTable[k] < minSup:
			            del (headerTable[k])
			
			    freqItemSet = set(headerTable.keys())
			
			    if len(freqItemSet) == 0:
			        return None, None
			
			    for k in headerTable:
			        headerTable[k] = [headerTable[k], None]
			    retTree = treeNode('Null Set', 1, None)
			
			    for tranSet, count in dataSet.items():
			        localD = {}
			        for item in tranSet:
			            if item in freqItemSet:
			                localD[item] = headerTable[item][0]
			        if len(localD) > 0:
			            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
			            updateTree(orderedItems, retTree, headerTable, count)
			    return retTree, headerTable
			
			
			def updateTree(items, inTree, headerTable, count):
			    """
			    使用频繁项集使FP树生长
			    :param items:
			    :param inTree:
			    :param headerTable:
			    :param count:
			    :return:
			    """
			    if items[0] in inTree.children:
			        inTree.children[items[0]].inc(count)
			    else:
			        inTree.children[items[0]] = treeNode(items[0], count, inTree)
			        if headerTable[items[0]][1] is None:
			            headerTable[items[0]][1] = inTree.children[items[0]]
			        else:
			            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
			    if len(items) > 1:
			        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
			
			
			def updateHeader(nodeToTest, targetNode):
			    """
			    更新头指针表，确保节点链接指向树中该元素项的每一个实例
			    :param nodeToTest:
			    :param targetNode:
			    :return:
			    """
			    while nodeToTest.nodeLink is not None:
			        nodeToTest = nodeToTest.nodeLink
			    nodeToTest.nodeLink = targetNode
			
			
			# 挖掘频繁项集
			def ascendTree(leafNode, prefixPath):
			    if leafNode.parent is not None:
			        prefixPath.append(leafNode.name)
			        ascendTree(leafNode.parent, prefixPath)
			
			
			def findPrefixPath(basePat, treeNode):
			    condPats = {}
			    while treeNode is not None:
			        prefixPath = []
			        ascendTree(treeNode, prefixPath)
			        if len(prefixPath) > 1:
			            condPats[frozenset(prefixPath[1:])] = treeNode.count
			        treeNode = treeNode.nodeLink
			    return condPats
			
			
			def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
			    """
			    递归查找频繁项集
			    :param inTree:
			    :param headerTable:
			    :param minSup:
			    :param preFix:
			    :param freqItemList:
			    :return:
			    """
			    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
			    for basePat in bigL:
			        newFreqSet = preFix.copy()
			        newFreqSet.add(basePat)
			        freqItemList.append(newFreqSet)
			        condPathBases = findPrefixPath(basePat, headerTable[basePat][1])
			        myCondTree, myHead = createTree(condPathBases, minSup)
			        if myHead is not None:
			            print('conditional tree for:', newFreqSet)
			            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
			
			
			# 生成数据集
			def loadSimpDat():
			    simData = [
			        ['r', 'z', 'h', 'j', 'p'],
			        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
			        ['z'],
			        ['r', 'x', 'n', 'o', 's'],
			        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
			        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
			    ]
			    return simData
			
			
			def createInitSet(dataSet):
			    retDict = {}
			    for trans in dataSet:
			        retDict[frozenset(trans)] = 1
			    return retDict
			
			
			if __name__ == '__main__':
			    minSup = 3
			    simDat = loadSimpDat()
			    initSet = createInitSet(simDat)
			    myFPtree, myHeaderTab = createTree(initSet, minSup)
			    myFPtree.disp()
			
			    myFreqList = []
			    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
			    print(myFreqList)
			
			```
- 补充说明
	- 参考书《Python3数据分析与机器学习实战》
	- 具体数据集和代码可以查看我的GitHub,欢迎star或者fork