import pandas as pd

class MarkovBlanket:
    """
    Solve a Markov blanket based on the adjacency matrix of the graph, that is,
    a subset containing all useful information for inferring the target variable
    (1.parent nodes, 2.child nodes and 3.co-parent nodes of the target node)
    1. o   o         2.  @             3.  o    @
        \ /             / \                 \  /
         @             o   o                 o

    Args:
        nodes(list): names of all input nodes
        graphmtx(np.ndarray): the adjacency matrix of graph (ndarray)
        dfmtx(pd.DataFrame): the adjacency matrix of graph (dataframe)
        target(str): the name of target node
    """

    def __init__(self, node_list, graphmtx):
        self.nodes = node_list
        self.graphmtx = graphmtx
        self.dfmtx = pd.DataFrame(self.graphmtx,columns = self.nodes,
                                  index = self.nodes)
    
    def getParents(self, target)->set:
        """
        get parent nodes of target node

        Args:
            target(str): the name of target node
        Returns:
            Set of parent nodes
        """
        return set(self.dfmtx.loc[self.dfmtx[target] != 0].index)
    
    def getChildren(self, target)->set:
        """
        get child nodes of target node
        Args:
            target(str): the name of target node
        Returns:
            Set of child nodes
        """
        return set(self.dfmtx.loc[self.dfmtx.loc[target] != 0].index)
    
    def getMB(self, target)->set:
        """
        get the Markov Blanket of target node

        Args:
            target(str): the name of target node
        Returns:
            Set of child Markov Blanket
        """
        self.target_pa = self.getParents(target)   ## Parents of target
        self.target_ch = self.getChildren(target)  ## Children of target
        self.target_cp = set()                     ## Co-parents of target
        for item in self.target_ch:
            coparents = self.getParents(item)
            self.target_cp = self.target_cp | coparents
        self.markovblanket = self.target_pa | self.target_ch | self.target_cp
        try:
            self.markovblanket.remove(target)
        except:
            pass
        return self.markovblanket