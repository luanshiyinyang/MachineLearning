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
