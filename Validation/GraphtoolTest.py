from graph_tool.all import *
import numpy as np

#G=load_graph("/Users/ksalamatian/processed.8.graphml")

g1=Graph()
g1.add_vertex(5000)
add_random_edges(g1,np.log(5000),parallel=False, self_loops=False)
prop = g1.new_vertex_property("int")
prop.a=np.ones((1,5000))
g2=Graph()
g2.add_vertex(5000)
add_random_edges(g2,np.log(5000),parallel=False, self_loops=False)
prop = g2.new_vertex_property("int")
prop.a=np.ones((1,5000))*2
g3=graph_union(g1,g2)
pos = sfdp_layout(g3)
graph_draw(g3, pos=pos, adjust_aspect=False, output="graph_original.pdf")
