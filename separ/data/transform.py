
from separ.data.sdp import SDP 
from separ.structs import Arc 
from separ.data.conll import EnhancedCoNLL

def sdp_to_conll(data: SDP, postprocess: bool = False) -> EnhancedCoNLL:
    news = []
    for graph in data:
        nodes = [EnhancedCoNLL.Node(node.ID, node.FORM, node.LEMMA, node.POS, '_', '_', i, '_', '_', '_') for i, node in enumerate(graph.nodes)]
        DEPS = [[] for _ in nodes]
        arcs = graph.arcs
        for arc in sorted(arcs):
            DEPS[arc.DEP-1].append(f'{arc.HEAD}:{arc.REL}')
        for i, (node, deps) in enumerate(zip(nodes, DEPS)):
            if len(deps) == 0:
                if postprocess:
                    node.DEPS = f'{i}:_'
                    arcs.append(Arc(i, i+1, '_'))
                else:
                    node.DEPS = f'_'
            else:
                assert len(deps) > 0
                node.DEPS = '|'.join(deps)
            
        new = EnhancedCoNLL.Graph(nodes, arcs, graph.annotations)
        news.append(new)
    return EnhancedCoNLL(news, data.name)

def conll_to_sdp(data: EnhancedCoNLL) -> SDP:
    news = []
    for graph in data:
        nodes = [SDP.Node(node.ID, node.FORM, node.LEMMA, node.UPOS, '-', '-', '_') for node in graph.nodes]
        # get only the first commented line 
        annotations, comment = [], False 
        for line in graph.annotations:
            if isinstance(line, int) or not comment:
                annotations.append(line)
                comment = True 
        new = SDP.Graph(nodes, [], annotations).rebuild(graph.arcs)
        news.append(new)
    return SDP(news, data.name)


