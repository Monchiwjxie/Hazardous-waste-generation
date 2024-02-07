import pandas as pd

class MarkovBlanket:
    def __init__(self, node_list, graphmtx):
        self.nodes = node_list
        self.graphmtx = graphmtx
        self.dfmtx = pd.DataFrame(self.graphmtx,columns = self.nodes,
                                  index = self.nodes)
    
    def getParents(self, target)->set:
        return set(self.dfmtx.loc[self.dfmtx[target] != 0].index)
    
    def getChildren(self, target)->set:
        return set(self.dfmtx.loc[self.dfmtx.loc[target] != 0].index)
    
    def getMB(self, target)->set:
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